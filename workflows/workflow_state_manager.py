"""
Unified Workflow State Manager

This module provides a unified interface for workflow state management,
combining state validation, checkpoint persistence, and audit logging.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .state_management import (
    CompleteWorkflowState, 
    WorkflowStateValidator,
    create_initial_workflow_state,
    update_workflow_state
)
from .checkpoint_manager import WorkflowCheckpointManager, AsyncWorkflowCheckpointManager
from .audit_logger import WorkflowAuditLogger, AuditEventType


logger = logging.getLogger(__name__)


class WorkflowStateManager:
    """
    Unified workflow state manager that coordinates state management,
    checkpointing, and audit logging for LangGraph workflows.
    """
    
    def __init__(self, 
                 checkpoint_db_path: str = "checkpoints/workflow_checkpoints.db",
                 audit_log_dir: str = "logs/audit",
                 auto_checkpoint: bool = True,
                 checkpoint_interval_steps: int = 1):
        """
        Initialize the workflow state manager.
        
        Args:
            checkpoint_db_path: Path to checkpoint database
            audit_log_dir: Directory for audit logs
            auto_checkpoint: Whether to automatically save checkpoints
            checkpoint_interval_steps: Steps between automatic checkpoints
        """
        self.checkpoint_manager = WorkflowCheckpointManager(checkpoint_db_path)
        self.audit_logger = WorkflowAuditLogger(audit_log_dir)
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval_steps = checkpoint_interval_steps
        
        # Track active workflows
        self._active_workflows: Dict[str, CompleteWorkflowState] = {}
        self._step_counters: Dict[str, int] = {}
        self._step_start_times: Dict[str, float] = {}
        
        logger.info("Initialized WorkflowStateManager")
    
    def create_workflow(self, suppliers: List[Dict[str, Any]], 
                       analysis_config: Dict[str, Any]) -> CompleteWorkflowState:
        """
        Create a new workflow with initial state.
        
        Args:
            suppliers: List of supplier dictionaries
            analysis_config: Configuration for the analysis
            
        Returns:
            CompleteWorkflowState: Initial workflow state
        """
        # Create initial state
        state = create_initial_workflow_state(suppliers, analysis_config)
        workflow_id = state["workflow_id"]
        
        # Store in active workflows
        self._active_workflows[workflow_id] = state
        self._step_counters[workflow_id] = 0
        
        # Log workflow start
        self.audit_logger.log_workflow_started(
            workflow_id=workflow_id,
            suppliers_count=len(suppliers),
            config=analysis_config
        )
        
        # Save initial checkpoint
        if self.auto_checkpoint:
            try:
                checkpoint_id = self.checkpoint_manager.save_checkpoint(
                    workflow_id=workflow_id,
                    state=state,
                    metadata={"step": "initial", "created_at": datetime.now().isoformat()}
                )
                self.audit_logger.log_checkpoint_saved(workflow_id, checkpoint_id, "initialize_workflow")
            except Exception as e:
                logger.error(f"Failed to save initial checkpoint for {workflow_id}: {e}")
        
        logger.info(f"Created new workflow: {workflow_id}")
        return state
    
    def update_state(self, workflow_id: str, updates: Dict[str, Any],
                    validate: bool = True, 
                    save_checkpoint: bool = None) -> CompleteWorkflowState:
        """
        Update workflow state with validation and optional checkpointing.
        
        Args:
            workflow_id: Workflow identifier
            updates: State updates to apply
            validate: Whether to validate state transition
            save_checkpoint: Whether to save checkpoint (None = use auto_checkpoint)
            
        Returns:
            CompleteWorkflowState: Updated state
            
        Raises:
            ValueError: If workflow not found or validation fails
        """
        if workflow_id not in self._active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found in active workflows")
        
        current_state = self._active_workflows[workflow_id]
        old_step = current_state.get("current_step")
        
        # Record step start time if transitioning to new step
        new_step = updates.get("current_step")
        if new_step and new_step != old_step:
            self._step_start_times[f"{workflow_id}_{new_step}"] = time.time()
        
        try:
            # Update state with validation
            updated_state = update_workflow_state(current_state, updates, validate)
            
            # Store updated state
            self._active_workflows[workflow_id] = updated_state
            self._step_counters[workflow_id] += 1
            
            # Log state transition if step changed
            if new_step and new_step != old_step:
                step_key = f"{workflow_id}_{old_step}"
                duration_ms = 0
                if step_key in self._step_start_times:
                    duration_ms = (time.time() - self._step_start_times[step_key]) * 1000
                    del self._step_start_times[step_key]
                
                self.audit_logger.log_state_transition(
                    workflow_id=workflow_id,
                    from_step=old_step,
                    to_step=new_step,
                    duration_ms=duration_ms,
                    metadata=updates.get("checkpoint_metadata", {})
                )
            
            # Save checkpoint if needed
            should_checkpoint = (
                save_checkpoint if save_checkpoint is not None 
                else (self.auto_checkpoint and 
                     self._step_counters[workflow_id] % self.checkpoint_interval_steps == 0)
            )
            
            if should_checkpoint:
                try:
                    checkpoint_id = self.checkpoint_manager.save_checkpoint(
                        workflow_id=workflow_id,
                        state=updated_state,
                        metadata={
                            "step": updated_state.get("current_step"),
                            "updated_at": datetime.now().isoformat(),
                            "step_counter": self._step_counters[workflow_id]
                        }
                    )
                    self.audit_logger.log_checkpoint_saved(
                        workflow_id, checkpoint_id, updated_state.get("current_step")
                    )
                except Exception as e:
                    logger.error(f"Failed to save checkpoint for {workflow_id}: {e}")
                    self.audit_logger.log_error(
                        workflow_id=workflow_id,
                        step=updated_state.get("current_step"),
                        error_message=str(e),
                        error_type="checkpoint_error",
                        retry_count=0
                    )
            
            return updated_state
            
        except ValueError as e:
            # Log validation error
            self.audit_logger.log_error(
                workflow_id=workflow_id,
                step=current_state.get("current_step"),
                error_message=str(e),
                error_type="validation_error",
                retry_count=current_state.get("retry_count", 0)
            )
            raise
    
    def get_state(self, workflow_id: str) -> Optional[CompleteWorkflowState]:
        """
        Get current workflow state.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            CompleteWorkflowState or None: Current state or None if not found
        """
        return self._active_workflows.get(workflow_id)
    
    def load_workflow(self, workflow_id: str, 
                     checkpoint_id: Optional[str] = None) -> Optional[CompleteWorkflowState]:
        """
        Load workflow from checkpoint.
        
        Args:
            workflow_id: Workflow identifier
            checkpoint_id: Specific checkpoint ID (None for latest)
            
        Returns:
            CompleteWorkflowState or None: Loaded state or None if not found
        """
        try:
            state = self.checkpoint_manager.load_checkpoint(workflow_id, checkpoint_id)
            
            if state:
                # Add to active workflows
                self._active_workflows[workflow_id] = state
                self._step_counters[workflow_id] = 0  # Reset counter
                
                # Log checkpoint load
                self.audit_logger.log_checkpoint_loaded(
                    workflow_id, checkpoint_id or "latest", state.get("current_step")
                )
                
                logger.info(f"Loaded workflow {workflow_id} from checkpoint")
                return state
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load workflow {workflow_id}: {e}")
            self.audit_logger.log_error(
                workflow_id=workflow_id,
                step="load_checkpoint",
                error_message=str(e),
                error_type="checkpoint_load_error",
                retry_count=0
            )
            return None
    
    def handle_error(self, workflow_id: str, error_message: str, 
                    error_type: str, max_retries: int = 3) -> CompleteWorkflowState:
        """
        Handle workflow error with retry logic.
        
        Args:
            workflow_id: Workflow identifier
            error_message: Error description
            error_type: Type of error
            max_retries: Maximum retry attempts
            
        Returns:
            CompleteWorkflowState: Updated state with error handling
        """
        if workflow_id not in self._active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        current_state = self._active_workflows[workflow_id]
        retry_count = current_state.get("retry_count", 0) + 1
        current_step = current_state.get("current_step")
        
        # Log error
        self.audit_logger.log_error(
            workflow_id=workflow_id,
            step=current_step,
            error_message=error_message,
            error_type=error_type,
            retry_count=retry_count
        )
        
        # Update state with error information
        error_updates = {
            "error_message": error_message,
            "has_errors": True,
            "retry_count": retry_count,
            "failed_steps": current_state.get("failed_steps", []) + [current_step]
        }
        
        # Determine if we should retry or fail
        if retry_count <= max_retries:
            # Log retry attempt
            self.audit_logger.log_retry_attempt(
                workflow_id=workflow_id,
                step=current_step,
                retry_count=retry_count,
                max_retries=max_retries
            )
            
            # Keep current step for retry
            error_updates["current_step"] = current_step
        else:
            # Max retries exceeded, mark as failed
            error_updates["current_step"] = "error_handling"
            error_updates["analysis_complete"] = False
            
            # Log workflow failure
            total_duration = (datetime.now() - current_state.get("started_at")).total_seconds() * 1000
            self.audit_logger.log_workflow_failed(
                workflow_id=workflow_id,
                step=current_step,
                error_message=error_message,
                total_duration_ms=total_duration,
                retry_count=retry_count
            )
        
        # Update state
        updated_state = self.update_state(
            workflow_id=workflow_id,
            updates=error_updates,
            validate=False,  # Skip validation for error states
            save_checkpoint=True  # Always save checkpoint on error
        )
        
        return updated_state
    
    def complete_workflow(self, workflow_id: str, 
                         final_report: Dict[str, Any]) -> CompleteWorkflowState:
        """
        Mark workflow as completed.
        
        Args:
            workflow_id: Workflow identifier
            final_report: Final analysis report
            
        Returns:
            CompleteWorkflowState: Final state
        """
        if workflow_id not in self._active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        current_state = self._active_workflows[workflow_id]
        
        # Calculate total duration
        total_duration = (datetime.now() - current_state.get("started_at")).total_seconds() * 1000
        
        # Update state to completed
        completion_updates = {
            "current_step": "completed",
            "analysis_complete": True,
            "final_report": final_report,
            "has_errors": False,  # Clear errors on successful completion
            "processing_times": current_state.get("processing_times", {})
        }
        completion_updates["processing_times"]["total_duration_ms"] = total_duration
        
        # Update state
        updated_state = self.update_state(
            workflow_id=workflow_id,
            updates=completion_updates,
            save_checkpoint=True  # Always save final checkpoint
        )
        
        # Log completion
        self.audit_logger.log_workflow_completed(
            workflow_id=workflow_id,
            total_duration_ms=total_duration,
            final_report=final_report
        )
        
        # Remove from active workflows
        del self._active_workflows[workflow_id]
        if workflow_id in self._step_counters:
            del self._step_counters[workflow_id]
        
        logger.info(f"Completed workflow {workflow_id} in {total_duration:.2f}ms")
        return updated_state
    
    def get_workflow_recovery_info(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get recovery information for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Dict or None: Recovery information
        """
        return self.checkpoint_manager.get_workflow_recovery_info(workflow_id)
    
    def list_active_workflows(self) -> List[str]:
        """Get list of active workflow IDs."""
        return list(self._active_workflows.keys())
    
    def get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get audit summary for a workflow."""
        return self.audit_logger.get_workflow_summary(workflow_id)
    
    def export_workflow_audit(self, workflow_id: str, 
                            output_file: Optional[str] = None) -> str:
        """Export workflow audit data."""
        return self.audit_logger.export_workflow_audit(workflow_id, output_file)
    
    def cleanup_completed_workflows(self, days_old: int = 7) -> int:
        """
        Clean up old completed workflows.
        
        Args:
            days_old: Days to keep completed workflows
            
        Returns:
            int: Number of workflows cleaned up
        """
        return self.checkpoint_manager.cleanup_old_checkpoints(days_old)
    
    def validate_workflow_state(self, workflow_id: str) -> Tuple[bool, List[str]]:
        """
        Validate current workflow state.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        if workflow_id not in self._active_workflows:
            return False, [f"Workflow {workflow_id} not found"]
        
        state = self._active_workflows[workflow_id]
        return WorkflowStateValidator.validate_data_integrity(state)
    
    def get_checkpointer(self):
        """Get the underlying checkpointer for LangGraph compilation."""
        return self.checkpoint_manager.get_checkpointer()


class AsyncWorkflowStateManager:
    """
    Asynchronous version of the workflow state manager.
    """
    
    def __init__(self, 
                 checkpoint_db_path: str = "checkpoints/async_workflow_checkpoints.db",
                 audit_log_dir: str = "logs/audit",
                 auto_checkpoint: bool = True,
                 checkpoint_interval_steps: int = 1):
        """Initialize async workflow state manager."""
        self.checkpoint_manager = AsyncWorkflowCheckpointManager(checkpoint_db_path)
        self.audit_logger = WorkflowAuditLogger(audit_log_dir)
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval_steps = checkpoint_interval_steps
        
        # Track active workflows
        self._active_workflows: Dict[str, CompleteWorkflowState] = {}
        self._step_counters: Dict[str, int] = {}
        self._step_start_times: Dict[str, float] = {}
        
        logger.info("Initialized AsyncWorkflowStateManager")
    
    async def get_checkpointer(self):
        """Get the underlying async checkpointer for LangGraph compilation."""
        return await self.checkpoint_manager.get_checkpointer()
    
    # Additional async methods would follow the same pattern as the sync version
    # but with async/await keywords where appropriate