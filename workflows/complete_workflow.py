"""
Complete Supply Chain Intelligence Workflow

This module implements the complete LangGraph workflow orchestration
for the supply chain intelligence platform, integrating all nodes
with conditional routing and error handling.

Requirements covered: 5.3, 5.4, 5.5, 5.6
"""

import asyncio
import logging
import math
import random
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state_management import CompleteWorkflowState, create_initial_workflow_state
from .workflow_state_manager import WorkflowStateManager
from .workflow_nodes import SupplyChainWorkflowNodes
from models.supplier import Supplier


logger = logging.getLogger(__name__)


class ExponentialBackoffCalculator:
    """
    Utility class for calculating exponential backoff delays with jitter.
    
    Implements exponential backoff with jitter to prevent thundering herd problems
    and provide more resilient retry mechanisms.
    """
    
    @staticmethod
    def calculate_delay(retry_count: int, base_delay: float = 1.0, 
                       max_delay: float = 300.0, jitter: bool = True) -> float:
        """
        Calculate exponential backoff delay with optional jitter.
        
        Args:
            retry_count: Current retry attempt (0-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Whether to add random jitter
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential delay: base_delay * 2^retry_count
        delay = base_delay * (2 ** retry_count)
        
        # Cap at maximum delay
        delay = min(delay, max_delay)
        
        # Add jitter to prevent thundering herd
        if jitter:
            # Add random jitter of ±25%
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            
            # Ensure delay is not negative
            delay = max(delay, 0.1)
        
        return delay
    
    @staticmethod
    async def wait_with_backoff(retry_count: int, base_delay: float = 1.0,
                               max_delay: float = 300.0, jitter: bool = True):
        """
        Wait for exponential backoff delay.
        
        Args:
            retry_count: Current retry attempt (0-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Whether to add random jitter
        """
        delay = ExponentialBackoffCalculator.calculate_delay(
            retry_count, base_delay, max_delay, jitter
        )
        
        logger.info(f"Waiting {delay:.2f} seconds before retry attempt {retry_count + 1}")
        await asyncio.sleep(delay)


class RetryConfiguration:
    """
    Configuration for retry behavior per workflow step.
    """
    
    # Step-specific retry configurations
    STEP_CONFIGS = {
        "initialize_workflow": {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "critical": True
        },
        "collect_web_data": {
            "max_retries": 5,
            "base_delay": 2.0,
            "max_delay": 120.0,
            "critical": False
        },
        "collect_api_data": {
            "max_retries": 4,
            "base_delay": 1.5,
            "max_delay": 60.0,
            "critical": False
        },
        "train_ml_models": {
            "max_retries": 2,
            "base_delay": 5.0,
            "max_delay": 300.0,
            "critical": False
        },
        "generate_ml_predictions": {
            "max_retries": 3,
            "base_delay": 2.0,
            "max_delay": 60.0,
            "critical": False
        },
        "run_crew_analysis": {
            "max_retries": 2,
            "base_delay": 10.0,
            "max_delay": 600.0,
            "critical": False
        },
        "validate_and_structure": {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "critical": False
        },
        "generate_final_report": {
            "max_retries": 3,
            "base_delay": 2.0,
            "max_delay": 60.0,
            "critical": True
        }
    }
    
    @classmethod
    def get_config(cls, step: str) -> Dict[str, Any]:
        """Get retry configuration for a specific step."""
        return cls.STEP_CONFIGS.get(step, {
            "max_retries": 3,
            "base_delay": 2.0,
            "max_delay": 60.0,
            "critical": False
        })
    
    @classmethod
    def should_retry(cls, step: str, retry_count: int) -> bool:
        """Check if step should be retried based on configuration."""
        config = cls.get_config(step)
        return retry_count < config["max_retries"]
    
    @classmethod
    def is_critical_step(cls, step: str) -> bool:
        """Check if step is critical for workflow success."""
        config = cls.get_config(step)
        return config.get("critical", False)


class CompleteSupplyChainWorkflow:
    """
    Complete workflow orchestration for supply chain intelligence analysis.
    
    Implements LangGraph StateGraph with conditional routing, error handling,
    and checkpoint-based persistence.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 checkpoint_db_path: str = "checkpoints/workflow_checkpoints.db"):
        """
        Initialize the complete workflow.
        
        Args:
            api_key: OpenAI API key for LLM operations
            checkpoint_db_path: Path to checkpoint database
        """
        self.api_key = api_key
        self.checkpoint_db_path = checkpoint_db_path
        
        # Initialize components
        self.state_manager = WorkflowStateManager(checkpoint_db_path)
        self.workflow_nodes = SupplyChainWorkflowNodes(api_key, self.state_manager)
        
        # Build the workflow graph
        self.workflow_graph = self._build_workflow_graph()
        
        logger.info("Initialized CompleteSupplyChainWorkflow")
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the complete workflow graph with all nodes and routing.
        
        Returns:
            Configured StateGraph for workflow execution
        """
        # Create the state graph
        workflow = StateGraph(CompleteWorkflowState)
        
        # Add all workflow nodes
        workflow.add_node("initialize_workflow", self.workflow_nodes.initialize_workflow)
        workflow.add_node("collect_web_data", self.workflow_nodes.collect_web_data)
        workflow.add_node("collect_api_data", self.workflow_nodes.collect_api_data)
        workflow.add_node("train_ml_models", self.workflow_nodes.train_ml_models)
        workflow.add_node("generate_ml_predictions", self.workflow_nodes.generate_ml_predictions)
        workflow.add_node("run_crew_analysis", self.workflow_nodes.run_crew_analysis)
        workflow.add_node("validate_and_structure", self.workflow_nodes.validate_and_structure)
        workflow.add_node("generate_final_report", self.workflow_nodes.generate_final_report)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("initialize_workflow")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "initialize_workflow",
            self._initialization_router,
            {
                "success": "collect_web_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "collect_web_data",
            self._data_collection_router,
            {
                "success": "collect_api_data",
                "retry": "collect_web_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "collect_api_data",
            self._data_collection_router,
            {
                "success": "train_ml_models",
                "retry": "collect_api_data", 
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "train_ml_models",
            self._ml_training_router,
            {
                "success": "generate_ml_predictions",
                "retry": "train_ml_models",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_ml_predictions",
            self._ml_prediction_router,
            {
                "success": "run_crew_analysis",
                "retry": "generate_ml_predictions",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "run_crew_analysis",
            self._crew_analysis_router,
            {
                "success": "validate_and_structure",
                "retry": "run_crew_analysis",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_and_structure",
            self._validation_router,
            {
                "success": "generate_final_report",
                "retry": "validate_and_structure",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_final_report",
            self._final_report_router,
            {
                "success": END,
                "retry": "generate_final_report",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_error",
            self._error_recovery_router,
            {
                "retry": "initialize_workflow",
                "partial_success": "generate_final_report",
                "terminal_failure": END
            }
        )
        
        # Compile with checkpointer
        checkpointer = self.state_manager.get_checkpointer()
        compiled_workflow = workflow.compile(checkpointer=checkpointer)
        
        logger.info("Workflow graph built and compiled successfully")
        return compiled_workflow
    
    # Router functions for conditional routing
    
    def _initialization_router(self, state: CompleteWorkflowState) -> Literal["success", "error"]:
        """Route after initialization step."""
        if state.get("initialization_complete") and not state.get("has_errors"):
            return "success"
        return "error"
    
    def _data_collection_router(self, state: CompleteWorkflowState) -> Literal["success", "retry", "error"]:
        """Route after data collection steps with enhanced retry logic."""
        current_step = state.get("current_step")
        retry_count = state.get("retry_count", 0)
        has_errors = state.get("has_errors", False)
        
        if not has_errors:
            return "success"
        elif RetryConfiguration.should_retry(current_step, retry_count):
            logger.info(f"Scheduling retry for {current_step}: attempt {retry_count + 1}")
            return "retry"
        else:
            logger.warning(f"Max retries exceeded for {current_step}: {retry_count} attempts")
            return "error"
    
    def _ml_training_router(self, state: CompleteWorkflowState) -> Literal["success", "retry", "error"]:
        """Route after ML training step with fallback support."""
        current_step = state.get("current_step")
        retry_count = state.get("retry_count", 0)
        has_errors = state.get("has_errors", False)
        
        # ML training can continue with rule-based approach even if training fails
        if not has_errors or state.get("ml_models_metadata", {}).get("approach") == "rule_based_fallback":
            return "success"
        elif RetryConfiguration.should_retry(current_step, retry_count):
            logger.info(f"Scheduling retry for {current_step}: attempt {retry_count + 1}")
            return "retry"
        else:
            logger.warning(f"ML training failed, falling back to rule-based approach")
            return "error"
    
    def _ml_prediction_router(self, state: CompleteWorkflowState) -> Literal["success", "retry", "error"]:
        """Route after ML prediction step with prediction validation."""
        current_step = state.get("current_step")
        retry_count = state.get("retry_count", 0)
        has_errors = state.get("has_errors", False)
        predictions = state.get("ml_predictions", [])
        
        if not has_errors and predictions:
            return "success"
        elif RetryConfiguration.should_retry(current_step, retry_count):
            logger.info(f"Scheduling retry for {current_step}: attempt {retry_count + 1}")
            return "retry"
        else:
            logger.warning(f"ML prediction generation failed after {retry_count} attempts")
            return "error"
    
    def _crew_analysis_router(self, state: CompleteWorkflowState) -> Literal["success", "retry", "error"]:
        """Route after crew analysis step with optional execution."""
        current_step = state.get("current_step")
        retry_count = state.get("retry_count", 0)
        has_errors = state.get("has_errors", False)
        crew_analysis = state.get("crew_analysis", {})
        
        # Crew analysis is optional - can continue without it
        if not has_errors or crew_analysis.get("status") == "skipped":
            return "success"
        elif RetryConfiguration.should_retry(current_step, retry_count):
            logger.info(f"Scheduling retry for {current_step}: attempt {retry_count + 1}")
            return "retry"
        else:
            logger.warning(f"Crew analysis failed, continuing without multi-agent insights")
            return "error"
    
    def _validation_router(self, state: CompleteWorkflowState) -> Literal["success", "retry", "error"]:
        """Route after validation step with data integrity checks."""
        current_step = state.get("current_step")
        retry_count = state.get("retry_count", 0)
        has_errors = state.get("has_errors", False)
        
        if not has_errors:
            return "success"
        elif RetryConfiguration.should_retry(current_step, retry_count):
            logger.info(f"Scheduling retry for {current_step}: attempt {retry_count + 1}")
            return "retry"
        else:
            logger.warning(f"Validation failed after {retry_count} attempts")
            return "error"
    
    def _final_report_router(self, state: CompleteWorkflowState) -> Literal["success", "retry", "error"]:
        """Route after final report generation with completion validation."""
        current_step = state.get("current_step")
        retry_count = state.get("retry_count", 0)
        has_errors = state.get("has_errors", False)
        final_report = state.get("final_report")
        
        if not has_errors and final_report:
            return "success"
        elif RetryConfiguration.should_retry(current_step, retry_count):
            logger.info(f"Scheduling retry for {current_step}: attempt {retry_count + 1}")
            return "retry"
        else:
            logger.error(f"Final report generation failed after {retry_count} attempts")
            return "error"
    
    def _error_recovery_router(self, state: CompleteWorkflowState) -> Literal["retry", "partial_success", "terminal_failure"]:
        """Route for error recovery decisions with enhanced logic."""
        retry_count = state.get("retry_count", 0)
        failed_steps = state.get("failed_steps", [])
        current_step = state.get("current_step", "unknown")
        
        # Check if we have partial results that can be used
        has_partial_results = (
            len(state.get("scraped_data", [])) > 0 or 
            len(state.get("ml_predictions", [])) > 0 or 
            state.get("crew_analysis", {}).get("result") or
            len(state.get("suppliers", [])) > 0
        )
        
        # Check for critical step failures
        critical_step_failed = any(
            RetryConfiguration.is_critical_step(step) for step in failed_steps
        )
        
        # Global retry limit reached
        if retry_count >= 5:
            logger.warning(f"Global retry limit reached for workflow {state['workflow_id']}")
            if has_partial_results:
                logger.info("Attempting partial success with available data")
                return "partial_success"
            else:
                logger.error("No partial results available, terminal failure")
                return "terminal_failure"
        
        # Critical step failures with high retry count
        if critical_step_failed and retry_count >= 3:
            logger.error(f"Critical step failure detected: {failed_steps}")
            return "terminal_failure"  # Critical failures should always terminate
        
        # Too many different steps have failed
        if len(set(failed_steps)) >= 4:
            logger.warning(f"Multiple step failures detected: {set(failed_steps)}")
            if has_partial_results:
                return "partial_success"
            else:
                return "terminal_failure"
        
        # Check step-specific retry limits
        step_config = RetryConfiguration.get_config(current_step)
        if retry_count >= step_config.get("max_retries", 3):
            logger.warning(f"Step-specific retry limit reached for {current_step}")
            if has_partial_results:
                return "partial_success"
            else:
                return "terminal_failure"
        
        # Otherwise, retry from the beginning with exponential backoff
        logger.info(f"Scheduling workflow retry: attempt {retry_count + 1}")
        return "retry"
    
    async def _handle_error(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Handle errors with graceful degradation, exponential backoff, and partial result preservation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with error handling
        """
        workflow_id = state["workflow_id"]
        current_step = state.get("current_step", "unknown")
        error_message = state.get("error_message", "Unknown error")
        retry_count = state.get("retry_count", 0)
        
        logger.error(f"Handling error in workflow {workflow_id} at step {current_step}: {error_message}")
        
        # Increment retry count
        new_retry_count = retry_count + 1
        
        # Preserve partial results
        partial_results = self._extract_partial_results(state)
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(state, new_retry_count)
        
        # Apply exponential backoff for retries
        if recovery_strategy == "retry":
            step_config = RetryConfiguration.get_config(current_step)
            
            # Calculate and apply exponential backoff
            await ExponentialBackoffCalculator.wait_with_backoff(
                retry_count=new_retry_count - 1,  # 0-based for calculation
                base_delay=step_config.get("base_delay", 2.0),
                max_delay=step_config.get("max_delay", 60.0),
                jitter=True
            )
            
            logger.info(f"Exponential backoff completed, retrying workflow {workflow_id}")
        
        # Update state based on recovery strategy
        if recovery_strategy == "retry":
            # Reset for retry with backoff applied
            updates = {
                "current_step": "initialize_workflow",
                "retry_count": new_retry_count,
                "has_errors": False,
                "error_message": None,
                "checkpoint_metadata": {
                    "recovery_attempt": new_retry_count,
                    "previous_error": error_message,
                    "recovery_strategy": "full_retry_with_backoff",
                    "backoff_applied": True,
                    "backoff_config": RetryConfiguration.get_config(current_step)
                }
            }
        elif recovery_strategy == "partial_success":
            # Generate report with partial results
            updates = {
                "current_step": "generate_final_report",
                "retry_count": new_retry_count,
                "has_errors": True,  # Keep error flag for reporting
                "checkpoint_metadata": {
                    "recovery_strategy": "partial_success",
                    "partial_results_used": True,
                    "partial_results_summary": partial_results
                }
            }
        else:  # terminal_failure
            # Create comprehensive failure report
            failure_report = {
                "workflow_id": workflow_id,
                "status": "terminal_failure",
                "error": error_message,
                "failed_at_step": current_step,
                "retry_count": new_retry_count,
                "failed_steps": state.get("failed_steps", []),
                "partial_results": partial_results,
                "error_analysis": {
                    "critical_step_failed": any(
                        RetryConfiguration.is_critical_step(step) 
                        for step in state.get("failed_steps", [])
                    ),
                    "multiple_failures": len(set(state.get("failed_steps", []))) >= 4,
                    "retry_limit_exceeded": new_retry_count >= 5
                },
                "generated_at": datetime.now().isoformat()
            }
            
            updates = {
                "current_step": "completed",
                "analysis_complete": False,
                "final_report": failure_report,
                "checkpoint_metadata": {
                    "recovery_strategy": "terminal_failure",
                    "final_status": "failed",
                    "failure_analysis": failure_report["error_analysis"]
                }
            }
        
        # Apply updates
        from .state_management import update_workflow_state
        updated_state = update_workflow_state(state, updates, validate=False)
        
        logger.info(f"Error handling completed for workflow {workflow_id}: strategy={recovery_strategy}")
        return updated_state
    
    def _extract_partial_results(self, state: CompleteWorkflowState) -> Dict[str, Any]:
        """Extract partial results from failed workflow."""
        return {
            "suppliers_analyzed": len(state.get("suppliers", [])),
            "web_data_collected": len(state.get("scraped_data", [])),
            "api_data_sources": len(state.get("api_data", {})),
            "ml_predictions_generated": len(state.get("ml_predictions", [])),
            "crew_analysis_available": bool(state.get("crew_analysis", {}).get("result")),
            "processing_times": state.get("processing_times", {}),
            "data_quality_scores": state.get("data_quality_scores", {})
        }
    
    def _determine_recovery_strategy(self, state: CompleteWorkflowState, 
                                   retry_count: int) -> Literal["retry", "partial_success", "terminal_failure"]:
        """
        Determine the appropriate recovery strategy based on enhanced criteria.
        
        Args:
            state: Current workflow state
            retry_count: Current retry count
            
        Returns:
            Recovery strategy to apply
        """
        failed_steps = state.get("failed_steps", [])
        current_step = state.get("current_step", "unknown")
        
        # Check for partial results with more granular assessment
        has_meaningful_data = (
            len(state.get("suppliers", [])) > 0 and
            (len(state.get("scraped_data", [])) > 0 or 
             len(state.get("ml_predictions", [])) > 0 or
             len(state.get("api_data", {})) > 0 or
             state.get("crew_analysis", {}).get("result"))
        )
        
        # Check data quality to determine if partial results are valuable
        data_quality_scores = state.get("data_quality_scores", {})
        avg_quality = sum(data_quality_scores.values()) / len(data_quality_scores) if data_quality_scores else 0.0
        has_quality_data = has_meaningful_data and avg_quality >= 0.3
        
        # Global retry limit reached
        if retry_count >= 5:
            logger.warning(f"Global retry limit reached: {retry_count} attempts")
            return "partial_success" if has_quality_data else "terminal_failure"
        
        # Critical step failures
        critical_failures = [step for step in failed_steps if RetryConfiguration.is_critical_step(step)]
        if critical_failures and retry_count >= 3:
            logger.error(f"Critical step failures detected: {critical_failures}")
            return "terminal_failure"
        
        # Multiple different step failures indicate systemic issues
        unique_failures = set(failed_steps)
        if len(unique_failures) >= 4:
            logger.warning(f"Multiple step failures detected: {unique_failures}")
            return "partial_success" if has_quality_data else "terminal_failure"
        
        # Step-specific retry limits
        step_config = RetryConfiguration.get_config(current_step)
        step_max_retries = step_config.get("max_retries", 3)
        
        if retry_count >= step_max_retries:
            logger.warning(f"Step-specific retry limit reached for {current_step}: {retry_count}/{step_max_retries}")
            if has_quality_data:
                return "partial_success"
            elif not RetryConfiguration.is_critical_step(current_step):
                # For non-critical steps, try partial success even with low quality data
                return "partial_success" if has_meaningful_data else "terminal_failure"
            else:
                return "terminal_failure"
        
        # Check for cascading failures (same step failing repeatedly)
        step_failure_count = failed_steps.count(current_step)
        if step_failure_count >= 3:
            logger.warning(f"Step {current_step} has failed {step_failure_count} times")
            return "partial_success" if has_quality_data else "terminal_failure"
        
        # Otherwise, retry with exponential backoff
        logger.info(f"Scheduling retry for workflow: attempt {retry_count}")
        return "retry"
    
    async def run_complete_analysis(self, suppliers: List[Supplier],
                                  analysis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete supply chain analysis workflow.
        
        Args:
            suppliers: List of suppliers to analyze
            analysis_config: Optional configuration for analysis
            
        Returns:
            Complete analysis results
        """
        # Convert suppliers to dictionaries if needed
        supplier_dicts = []
        for supplier in suppliers:
            if isinstance(supplier, Supplier):
                supplier_dicts.append(supplier.model_dump())
            else:
                supplier_dicts.append(supplier)
        
        # Create initial state
        initial_state = create_initial_workflow_state(
            suppliers=supplier_dicts,
            analysis_config=analysis_config or {}
        )
        
        workflow_id = initial_state["workflow_id"]
        logger.info(f"Starting complete analysis workflow {workflow_id} with {len(suppliers)} suppliers")
        
        try:
            # Configure workflow execution
            config = {
                "configurable": {
                    "thread_id": workflow_id
                }
            }
            
            # Execute workflow
            final_state = await self.workflow_graph.ainvoke(initial_state, config)
            
            # Extract results
            result = {
                "workflow_id": workflow_id,
                "success": final_state.get("analysis_complete", False),
                "final_report": final_state.get("final_report"),
                "execution_metadata": {
                    "total_duration_ms": sum(final_state.get("processing_times", {}).values()),
                    "steps_completed": self._get_completed_steps(final_state),
                    "steps_failed": final_state.get("failed_steps", []),
                    "retry_count": final_state.get("retry_count", 0),
                    "data_quality_scores": final_state.get("data_quality_scores", {})
                }
            }
            
            logger.info(f"Workflow {workflow_id} completed: success={result['success']}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} execution failed: {str(e)}")
            
            return {
                "workflow_id": workflow_id,
                "success": False,
                "error": str(e),
                "execution_metadata": {
                    "failed_during_execution": True
                }
            }
    
    def _get_completed_steps(self, state: CompleteWorkflowState) -> List[str]:
        """Get list of successfully completed steps."""
        completed = []
        
        if state.get("initialization_complete"):
            completed.append("initialize_workflow")
        if state.get("data_collection_status") in ["web_complete", "complete"]:
            completed.append("collect_web_data")
        if state.get("data_collection_status") in ["api_complete", "complete"]:
            completed.append("collect_api_data")
        if state.get("ml_training_complete"):
            completed.append("train_ml_models")
        if state.get("ml_predictions_complete"):
            completed.append("generate_ml_predictions")
        if state.get("crew_analysis_complete"):
            completed.append("run_crew_analysis")
        if state.get("validation_complete"):
            completed.append("validate_and_structure")
        if state.get("analysis_complete"):
            completed.append("generate_final_report")
        
        return completed
    
    async def resume_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Resume a workflow from its last checkpoint.
        
        Args:
            workflow_id: ID of workflow to resume
            
        Returns:
            Workflow execution results
        """
        logger.info(f"Resuming workflow {workflow_id}")
        
        try:
            # Load workflow state from checkpoint
            state = self.state_manager.load_workflow(workflow_id)
            
            if not state:
                return {
                    "workflow_id": workflow_id,
                    "success": False,
                    "error": "Workflow not found or no checkpoint available"
                }
            
            # Configure workflow execution
            config = {
                "configurable": {
                    "thread_id": workflow_id
                }
            }
            
            # Resume execution
            final_state = await self.workflow_graph.ainvoke(state, config)
            
            # Extract results
            result = {
                "workflow_id": workflow_id,
                "success": final_state.get("analysis_complete", False),
                "final_report": final_state.get("final_report"),
                "resumed": True,
                "execution_metadata": {
                    "total_duration_ms": sum(final_state.get("processing_times", {}).values()),
                    "steps_completed": self._get_completed_steps(final_state),
                    "steps_failed": final_state.get("failed_steps", []),
                    "retry_count": final_state.get("retry_count", 0)
                }
            }
            
            logger.info(f"Resumed workflow {workflow_id} completed: success={result['success']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to resume workflow {workflow_id}: {str(e)}")
            
            return {
                "workflow_id": workflow_id,
                "success": False,
                "error": f"Resume failed: {str(e)}",
                "resumed": False
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get current status of a workflow.
        
        Args:
            workflow_id: ID of workflow to check
            
        Returns:
            Workflow status information
        """
        # Try to get from active workflows first
        state = self.state_manager.get_state(workflow_id)
        
        if not state:
            # Try to load from checkpoint
            state = self.state_manager.load_workflow(workflow_id)
        
        if not state:
            return {
                "workflow_id": workflow_id,
                "status": "not_found",
                "error": "Workflow not found"
            }
        
        return {
            "workflow_id": workflow_id,
            "status": "running" if not state.get("analysis_complete") else "completed",
            "current_step": state.get("current_step"),
            "progress": {
                "initialization_complete": state.get("initialization_complete", False),
                "data_collection_status": state.get("data_collection_status", "pending"),
                "ml_training_complete": state.get("ml_training_complete", False),
                "ml_predictions_complete": state.get("ml_predictions_complete", False),
                "crew_analysis_complete": state.get("crew_analysis_complete", False),
                "validation_complete": state.get("validation_complete", False),
                "analysis_complete": state.get("analysis_complete", False)
            },
            "has_errors": state.get("has_errors", False),
            "retry_count": state.get("retry_count", 0),
            "failed_steps": state.get("failed_steps", []),
            "processing_times": state.get("processing_times", {}),
            "started_at": state.get("started_at"),
            "updated_at": state.get("updated_at")
        }