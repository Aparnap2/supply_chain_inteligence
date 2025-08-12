"""
CrewAI Multi-Agent System Orchestrator

This module provides the main orchestrator for the CrewAI multi-agent system,
integrating all specialized agents, coordination, and task management.

Requirements covered: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from crewai import Crew, Process
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from datetime import datetime
from pathlib import Path

from .specialized_agents import SupplyChainAgents
from .coordination import AgentCoordinator
from .task_definitions import SupplyChainTaskDefinitions


class SupplyChainCrewOrchestrator:
    """
    Main orchestrator for the CrewAI multi-agent supply chain intelligence system
    
    This class integrates all components:
    - Specialized AI agents (Data Analysis, ML Specialist, Risk Assessment, Strategy)
    - Agent coordination and task management
    - Task definitions with structured outputs
    - Performance monitoring and error handling
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o",
        verbose: bool = True,
        enable_memory: bool = True,
        max_retries: int = 3,
        process_type: Process = Process.sequential
    ):
        """
        Initialize the crew orchestrator
        
        Args:
            llm_model: LLM model to use for all agents
            verbose: Enable verbose logging
            enable_memory: Enable long-term memory for agents
            max_retries: Maximum retry attempts for failed executions
            process_type: Execution process (sequential or hierarchical)
        """
        self.llm_model = llm_model
        self.verbose = verbose
        self.enable_memory = enable_memory
        self.max_retries = max_retries
        self.process_type = process_type
        
        # Initialize components
        self.agent_factory = SupplyChainAgents(llm_model, verbose)
        self.coordinator = AgentCoordinator(
            llm_model=llm_model,
            verbose=verbose,
            max_retries=max_retries,
            enable_memory=enable_memory
        )
        self.task_factory = SupplyChainTaskDefinitions(self.agent_factory)
        
        # Create the coordinated crew
        self.crew = self.coordinator.create_coordinated_crew(
            process_type=process_type,
            enable_planning=True
        )
        
        # Execution history
        self.execution_history = []
    
    def run_complete_analysis(
        self,
        suppliers_data: List[Dict[str, Any]],
        scraped_data: List[Dict[str, Any]],
        ml_predictions: List[Dict[str, Any]],
        model_metadata: Dict[str, Any],
        business_context: Dict[str, Any],
        save_results: bool = True,
        results_dir: str = "results"
    ) -> Dict[str, Any]:
        """
        Run complete supply chain intelligence analysis
        
        Args:
            suppliers_data: Supplier information and metrics
            scraped_data: Web-scraped content and news data
            ml_predictions: ML model predictions to validate
            model_metadata: Information about ML models used
            business_context: Business objectives and constraints
            save_results: Whether to save results to files
            results_dir: Directory to save results
            
        Returns:
            Complete analysis results with all agent outputs
        """
        execution_start = datetime.now()
        execution_id = f"analysis_{int(execution_start.timestamp())}"
        
        try:
            # Create complete workflow tasks
            tasks = self.task_factory.create_complete_analysis_workflow(
                suppliers_data=suppliers_data,
                scraped_data=scraped_data,
                ml_predictions=ml_predictions,
                model_metadata=model_metadata,
                business_context=business_context
            )
            
            # Prepare inputs for execution
            inputs = {
                "suppliers_data": suppliers_data,
                "scraped_data": scraped_data,
                "ml_predictions": ml_predictions,
                "model_metadata": model_metadata,
                "business_context": business_context,
                "execution_id": execution_id
            }
            
            # Execute with coordination
            result = self.coordinator.execute_with_coordination(
                crew=self.crew,
                tasks=tasks,
                inputs=inputs,
                context_sharing=True
            )
            
            # Process and structure results
            structured_result = self._process_execution_result(result, execution_id)
            
            # Save results if requested
            if save_results:
                self._save_results(structured_result, results_dir, execution_id)
            
            # Update execution history
            self.execution_history.append({
                "execution_id": execution_id,
                "timestamp": execution_start.isoformat(),
                "success": structured_result["success"],
                "execution_time": structured_result.get("execution_time", 0)
            })
            
            return structured_result
            
        except Exception as e:
            error_result = {
                "execution_id": execution_id,
                "success": False,
                "error": str(e),
                "timestamp": execution_start.isoformat(),
                "execution_time": (datetime.now() - execution_start).total_seconds()
            }
            
            # Update execution history
            self.execution_history.append({
                "execution_id": execution_id,
                "timestamp": execution_start.isoformat(),
                "success": False,
                "execution_time": error_result["execution_time"]
            })
            
            return error_result
    
    def run_individual_analysis(
        self,
        analysis_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run individual analysis component
        
        Args:
            analysis_type: Type of analysis ('data', 'ml', 'risk', 'strategy')
            **kwargs: Arguments specific to the analysis type
            
        Returns:
            Individual analysis results
        """
        execution_id = f"{analysis_type}_{int(datetime.now().timestamp())}"
        
        try:
            if analysis_type == "data":
                task = self.task_factory.create_data_analysis_task(
                    suppliers_data=kwargs.get("suppliers_data", []),
                    scraped_data=kwargs.get("scraped_data", []),
                    context_data=kwargs.get("context_data")
                )
            elif analysis_type == "ml":
                task = self.task_factory.create_ml_validation_task(
                    ml_predictions=kwargs.get("ml_predictions", []),
                    model_metadata=kwargs.get("model_metadata", {}),
                    validation_data=kwargs.get("validation_data")
                )
            elif analysis_type == "risk":
                task = self.task_factory.create_risk_assessment_task(
                    analysis_results=kwargs.get("analysis_results", {}),
                    ml_validation=kwargs.get("ml_validation", {}),
                    business_context=kwargs.get("business_context", {})
                )
            elif analysis_type == "strategy":
                task = self.task_factory.create_strategy_generation_task(
                    risk_assessment=kwargs.get("risk_assessment", {}),
                    business_objectives=kwargs.get("business_objectives", {}),
                    resource_constraints=kwargs.get("resource_constraints", {})
                )
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Execute single task
            result = self.coordinator.execute_with_coordination(
                crew=self.crew,
                tasks=[task],
                inputs=kwargs,
                context_sharing=False
            )
            
            return self._process_execution_result(result, execution_id)
            
        except Exception as e:
            return {
                "execution_id": execution_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_execution_result(
        self, 
        raw_result: Dict[str, Any], 
        execution_id: str
    ) -> Dict[str, Any]:
        """
        Process and structure execution results
        
        Args:
            raw_result: Raw result from crew execution
            execution_id: Unique execution identifier
            
        Returns:
            Structured result with parsed outputs
        """
        if not raw_result.get("success", False):
            return raw_result
        
        # Extract structured outputs from tasks
        crew_result = raw_result.get("result")
        structured_outputs = {}
        
        # Parse individual task outputs if available
        if hasattr(crew_result, 'tasks_output'):
            for i, task_output in enumerate(crew_result.tasks_output):
                task_name = f"task_{i}"
                
                # Try to extract Pydantic model output
                if hasattr(task_output, 'pydantic') and task_output.pydantic:
                    structured_outputs[task_name] = task_output.pydantic.model_dump()
                elif hasattr(task_output, 'json_dict') and task_output.json_dict:
                    structured_outputs[task_name] = task_output.json_dict
                else:
                    structured_outputs[task_name] = str(task_output)
        
        return {
            "execution_id": execution_id,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "execution_time": raw_result.get("execution_time", 0),
            "structured_outputs": structured_outputs,
            "raw_result": str(crew_result),
            "performance_metrics": raw_result.get("metrics", {}),
            "agent_roles": self.agent_factory.get_agent_roles()
        }
    
    def _save_results(
        self, 
        result: Dict[str, Any], 
        results_dir: str, 
        execution_id: str
    ):
        """
        Save execution results to files
        
        Args:
            result: Structured execution result
            results_dir: Directory to save results
            execution_id: Unique execution identifier
        """
        # Create results directory
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        # Save main result
        result_file = results_path / f"{execution_id}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Save individual structured outputs
        if "structured_outputs" in result:
            for task_name, output in result["structured_outputs"].items():
                output_file = results_path / f"{execution_id}_{task_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=2, default=str)
        
        # Save performance metrics
        if "performance_metrics" in result:
            metrics_file = results_path / f"{execution_id}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(result["performance_metrics"], f, indent=2, default=str)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all executions"""
        if not self.execution_history:
            return {"message": "No executions recorded"}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for exec in self.execution_history if exec["success"])
        failed_executions = total_executions - successful_executions
        
        execution_times = [exec["execution_time"] for exec in self.execution_history if "execution_time" in exec]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "average_execution_time": avg_execution_time,
            "coordinator_metrics": self.coordinator.get_performance_metrics()
        }
    
    def reset_performance_metrics(self):
        """Reset all performance metrics"""
        self.execution_history = []
        self.coordinator.reset_metrics()


# Convenience function for creating and running analysis
def run_supply_chain_analysis(
    suppliers_data: List[Dict[str, Any]],
    scraped_data: List[Dict[str, Any]],
    ml_predictions: List[Dict[str, Any]],
    model_metadata: Dict[str, Any],
    business_context: Dict[str, Any],
    llm_model: str = "gpt-4o",
    verbose: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run complete supply chain analysis
    
    Returns:
        Complete analysis results
    """
    orchestrator = SupplyChainCrewOrchestrator(
        llm_model=llm_model,
        verbose=verbose
    )
    
    return orchestrator.run_complete_analysis(
        suppliers_data=suppliers_data,
        scraped_data=scraped_data,
        ml_predictions=ml_predictions,
        model_metadata=model_metadata,
        business_context=business_context,
        save_results=save_results
    )