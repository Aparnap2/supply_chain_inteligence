"""
Agent Coordination and Task Management System

This module implements the coordination system for CrewAI multi-agent workflows,
including delegation settings, task execution management, and performance monitoring.

Requirements covered: 3.5
"""

from crewai import Crew, Process, Task
from crewai.memory import LongTermMemory
from typing import List, Dict, Any, Optional, Callable
import logging
import time
from datetime import datetime
import json
from pathlib import Path

from .specialized_agents import SupplyChainAgents


class AgentCoordinator:
    """
    Manages agent coordination, task delegation, and workflow execution
    
    Requirements: 3.5 - Implement conditional routing to error handling nodes 
    with retry logic and task dependency management
    """
    
    def __init__(
        self, 
        llm_model: str = "gpt-4o",
        verbose: bool = True,
        max_retries: int = 3,
        enable_memory: bool = True,
        log_file: Optional[str] = None
    ):
        """
        Initialize the agent coordinator
        
        Args:
            llm_model: LLM model to use for agents
            verbose: Enable verbose logging
            max_retries: Maximum retry attempts for failed tasks
            enable_memory: Enable long-term memory for agents
            log_file: Optional log file path for performance monitoring
        """
        self.llm_model = llm_model
        self.verbose = verbose
        self.max_retries = max_retries
        self.enable_memory = enable_memory
        
        # Initialize logging
        self.logger = self._setup_logging(log_file)
        
        # Initialize agent factory
        self.agent_factory = SupplyChainAgents(llm_model, verbose)
        
        # Performance tracking
        self.execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "agent_performance": {}
        }
        
        # Task dependency tracking
        self.task_dependencies = {}
        self.task_context = {}
    
    def _setup_logging(self, log_file: Optional[str]) -> logging.Logger:
        """Setup logging for agent performance monitoring"""
        logger = logging.getLogger("AgentCoordinator")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def create_coordinated_crew(
        self, 
        process_type: Process = Process.sequential,
        enable_planning: bool = True,
        memory_provider: Optional[str] = None
    ) -> Crew:
        """
        Create a coordinated crew with all specialized agents
        
        Args:
            process_type: Execution process (sequential or hierarchical)
            enable_planning: Enable dynamic planning capabilities
            memory_provider: Memory provider for long-term memory
            
        Returns:
            Configured CrewAI crew with coordination settings
        """
        # Get all specialized agents
        agents = self.agent_factory.get_all_agents()
        
        # Configure crew with coordination settings
        crew_config = {
            "agents": agents,
            "process": process_type,
            "verbose": self.verbose,
            "planning": enable_planning,
            "max_execution_time": 1800,  # 30 minutes timeout
            "step_callback": self._log_step_performance
        }
        
        # Add memory if enabled
        if self.enable_memory:
            crew_config["memory"] = True
            if memory_provider:
                crew_config["memory_provider"] = memory_provider
        
        # Add manager LLM for hierarchical process
        if process_type == Process.hierarchical:
            crew_config["manager_llm"] = self.llm_model
        
        self.logger.info(f"Created coordinated crew with {len(agents)} agents")
        return Crew(**crew_config)
    
    def execute_with_coordination(
        self, 
        crew: Crew, 
        tasks: List[Task],
        inputs: Dict[str, Any],
        context_sharing: bool = True
    ) -> Dict[str, Any]:
        """
        Execute crew tasks with coordination and error handling
        
        Args:
            crew: The CrewAI crew to execute
            tasks: List of tasks to execute
            inputs: Input parameters for task execution
            context_sharing: Enable context sharing between tasks
            
        Returns:
            Execution results with performance metrics
        """
        start_time = time.time()
        execution_id = f"exec_{int(start_time)}"
        
        self.logger.info(f"Starting coordinated execution {execution_id}")
        
        try:
            # Set up task dependencies and context sharing
            if context_sharing:
                self._setup_task_context(tasks, inputs)
            
            # Add tasks to crew
            crew.tasks = tasks
            
            # Execute with retry logic
            result = self._execute_with_retries(crew, inputs, execution_id)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_id, execution_time, True)
            
            self.logger.info(f"Execution {execution_id} completed successfully in {execution_time:.2f}s")
            
            return {
                "execution_id": execution_id,
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "metrics": self.get_performance_metrics()
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_id, execution_time, False)
            
            self.logger.error(f"Execution {execution_id} failed: {str(e)}")
            
            return {
                "execution_id": execution_id,
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "metrics": self.get_performance_metrics()
            }
    
    def _setup_task_context(self, tasks: List[Task], inputs: Dict[str, Any]):
        """
        Set up context sharing between tasks for sequential execution
        
        Args:
            tasks: List of tasks to configure
            inputs: Initial input context
        """
        # Initialize context with inputs
        self.task_context = inputs.copy()
        
        # Configure context parameter for each task
        for i, task in enumerate(tasks):
            if i > 0:  # Skip first task
                # Add context from previous tasks
                task.context = list(tasks[:i])
                self.logger.debug(f"Task {i} configured with context from {i} previous tasks")
    
    def _execute_with_retries(
        self, 
        crew: Crew, 
        inputs: Dict[str, Any], 
        execution_id: str
    ) -> Any:
        """
        Execute crew with retry logic for error handling
        
        Args:
            crew: The crew to execute
            inputs: Input parameters
            execution_id: Unique execution identifier
            
        Returns:
            Execution result
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"Execution {execution_id} attempt {attempt + 1}")
                
                # Execute the crew
                result = crew.kickoff(inputs=inputs)
                
                self.logger.info(f"Execution {execution_id} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Execution {execution_id} attempt {attempt + 1} failed: {str(e)}"
                )
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Execution {execution_id} failed after {self.max_retries + 1} attempts")
        
        # If we get here, all retries failed
        raise last_exception
    
    def _log_step_performance(self, step_output: Any):
        """
        Callback function to log individual step performance
        
        Args:
            step_output: Output from the completed step
        """
        timestamp = datetime.now().isoformat()
        
        # Extract agent information if available
        agent_role = getattr(step_output, 'agent', {}).get('role', 'Unknown')
        
        # Log step completion
        self.logger.debug(f"Step completed by {agent_role} at {timestamp}")
        
        # Update agent-specific performance metrics
        if agent_role not in self.execution_metrics["agent_performance"]:
            self.execution_metrics["agent_performance"][agent_role] = {
                "total_steps": 0,
                "successful_steps": 0,
                "average_step_time": 0.0
            }
        
        self.execution_metrics["agent_performance"][agent_role]["total_steps"] += 1
        self.execution_metrics["agent_performance"][agent_role]["successful_steps"] += 1
    
    def _update_performance_metrics(
        self, 
        execution_id: str, 
        execution_time: float, 
        success: bool
    ):
        """
        Update overall performance metrics
        
        Args:
            execution_id: Unique execution identifier
            execution_time: Time taken for execution
            success: Whether execution was successful
        """
        self.execution_metrics["total_executions"] += 1
        
        if success:
            self.execution_metrics["successful_executions"] += 1
        else:
            self.execution_metrics["failed_executions"] += 1
        
        # Update average execution time
        total_time = (
            self.execution_metrics["average_execution_time"] * 
            (self.execution_metrics["total_executions"] - 1) + 
            execution_time
        )
        self.execution_metrics["average_execution_time"] = (
            total_time / self.execution_metrics["total_executions"]
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.execution_metrics.copy()
    
    def save_performance_metrics(self, file_path: str):
        """
        Save performance metrics to file
        
        Args:
            file_path: Path to save metrics file
        """
        metrics_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.execution_metrics
        }
        
        with open(file_path, 'w') as f:
            json.dump(metrics_with_timestamp, f, indent=2)
        
        self.logger.info(f"Performance metrics saved to {file_path}")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "agent_performance": {}
        }
        self.logger.info("Performance metrics reset")


# Convenience function for creating a coordinated crew
def create_coordinated_crew(
    llm_model: str = "gpt-4o",
    process_type: Process = Process.sequential,
    verbose: bool = True,
    enable_memory: bool = True
) -> tuple[Crew, AgentCoordinator]:
    """
    Create a coordinated crew with default settings
    
    Returns:
        Tuple of (crew, coordinator) for easy access
    """
    coordinator = AgentCoordinator(
        llm_model=llm_model,
        verbose=verbose,
        enable_memory=enable_memory
    )
    
    crew = coordinator.create_coordinated_crew(
        process_type=process_type,
        enable_planning=True
    )
    
    return crew, coordinator