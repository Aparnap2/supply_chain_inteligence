"""
Workflow State Management for Supply Chain Intelligence Platform

This module implements the CompleteWorkflowState TypedDict and related state management
functionality for LangGraph workflow orchestration.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from typing_extensions import TypedDict

from models.analysis_result import AnalysisResult
from models.scraped_data import ScrapedData


class CompleteWorkflowState(TypedDict):
    """
    Complete workflow state for LangGraph orchestration.
    
    This TypedDict defines all the state fields required for the supply chain
    intelligence workflow, including input configuration, processing results,
    state tracking, and error handling.
    """
    
    # Input and configuration
    suppliers: List[Dict[str, Any]]
    analysis_config: Dict[str, Any]
    
    # State tracking and metadata
    current_step: str
    workflow_id: str
    started_at: datetime
    updated_at: datetime
    retry_count: int
    
    # Error handling
    error_message: Optional[str]
    has_errors: bool
    failed_steps: List[str]
    
    # Processing status flags
    initialization_complete: bool
    data_collection_status: Literal["pending", "web_complete", "api_complete", "complete", "web_error", "api_error", "failed"]
    ml_training_complete: bool
    ml_predictions_complete: bool
    crew_analysis_complete: bool
    validation_complete: bool
    analysis_complete: bool
    
    # Processing results
    scraped_data: List[Dict[str, Any]]
    api_data: Dict[str, Any]
    ml_models_metadata: Dict[str, Any]
    ml_predictions: List[Dict[str, Any]]
    crew_analysis: Dict[str, Any]
    validated_results: Optional[AnalysisResult]
    
    # Final output
    final_report: Optional[Dict[str, Any]]
    
    # Audit and metadata
    processing_times: Dict[str, float]
    data_quality_scores: Dict[str, float]
    checkpoint_metadata: Dict[str, Any]


class WorkflowStateValidator:
    """
    Validates workflow state transitions and data integrity.
    """
    
    REQUIRED_INITIAL_FIELDS = [
        "suppliers", "analysis_config", "workflow_id", 
        "current_step", "started_at", "updated_at"
    ]
    
    VALID_STEPS = [
        "initialize_workflow",
        "collect_web_data", 
        "collect_api_data",
        "train_ml_models",
        "generate_ml_predictions",
        "run_crew_analysis",
        "validate_and_structure",
        "generate_final_report",
        "error_handling",
        "completed"
    ]
    
    VALID_DATA_COLLECTION_STATUSES = [
        "pending", "web_complete", "api_complete", "complete", 
        "web_error", "api_error", "failed"
    ]
    
    @classmethod
    def validate_initial_state(cls, state: CompleteWorkflowState) -> tuple[bool, List[str]]:
        """
        Validate that the initial state has all required fields.
        
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in cls.REQUIRED_INITIAL_FIELDS:
            if field not in state or state[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate suppliers list
        if not isinstance(state.get("suppliers"), list) or len(state["suppliers"]) == 0:
            errors.append("Suppliers must be a non-empty list")
        
        # Validate workflow_id format
        workflow_id = state.get("workflow_id")
        if workflow_id:
            try:
                uuid.UUID(workflow_id)
            except (ValueError, TypeError):
                errors.append("workflow_id must be a valid UUID string")
        
        # Validate current_step
        if state.get("current_step") not in cls.VALID_STEPS:
            errors.append(f"current_step must be one of: {cls.VALID_STEPS}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def validate_state_transition(cls, old_state: CompleteWorkflowState, 
                                new_state: CompleteWorkflowState) -> tuple[bool, List[str]]:
        """
        Validate that a state transition is valid.
        
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        # Workflow ID should not change
        if old_state.get("workflow_id") != new_state.get("workflow_id"):
            errors.append("workflow_id cannot be changed during state transition")
        
        # Validate step progression
        old_step = old_state.get("current_step")
        new_step = new_state.get("current_step")
        
        if new_step not in cls.VALID_STEPS:
            errors.append(f"Invalid new step: {new_step}")
        
        # Validate data collection status
        data_status = new_state.get("data_collection_status")
        if data_status and data_status not in cls.VALID_DATA_COLLECTION_STATUSES:
            errors.append(f"Invalid data_collection_status: {data_status}")
        
        # updated_at should be more recent
        old_updated = old_state.get("updated_at")
        new_updated = new_state.get("updated_at")
        if old_updated and new_updated and new_updated <= old_updated:
            errors.append("updated_at must be more recent than previous state")
        
        return len(errors) == 0, errors
    
    @classmethod
    def validate_data_integrity(cls, state: CompleteWorkflowState) -> tuple[bool, List[str]]:
        """
        Validate data integrity within the state.
        
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        # If ml_predictions_complete is True, ml_predictions should not be empty
        if state.get("ml_predictions_complete") and not state.get("ml_predictions"):
            errors.append("ml_predictions_complete is True but ml_predictions is empty")
        
        # If crew_analysis_complete is True, crew_analysis should not be empty
        if state.get("crew_analysis_complete") and not state.get("crew_analysis"):
            errors.append("crew_analysis_complete is True but crew_analysis is empty")
        
        # If analysis_complete is True, final_report should exist
        if state.get("analysis_complete") and not state.get("final_report"):
            errors.append("analysis_complete is True but final_report is missing")
        
        # Retry count should not be negative
        retry_count = state.get("retry_count", 0)
        if retry_count < 0:
            errors.append("retry_count cannot be negative")
        
        return len(errors) == 0, errors


def create_initial_workflow_state(suppliers: List[Dict[str, Any]], 
                                analysis_config: Dict[str, Any]) -> CompleteWorkflowState:
    """
    Create an initial workflow state with default values.
    
    Args:
        suppliers: List of supplier dictionaries
        analysis_config: Configuration for the analysis
        
    Returns:
        CompleteWorkflowState: Initial state with all required fields
    """
    workflow_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    return CompleteWorkflowState(
        # Input and configuration
        suppliers=suppliers,
        analysis_config=analysis_config,
        
        # State tracking and metadata
        current_step="initialize_workflow",
        workflow_id=workflow_id,
        started_at=current_time,
        updated_at=current_time,
        retry_count=0,
        
        # Error handling
        error_message=None,
        has_errors=False,
        failed_steps=[],
        
        # Processing status flags
        initialization_complete=False,
        data_collection_status="pending",
        ml_training_complete=False,
        ml_predictions_complete=False,
        crew_analysis_complete=False,
        validation_complete=False,
        analysis_complete=False,
        
        # Processing results
        scraped_data=[],
        api_data={},
        ml_models_metadata={},
        ml_predictions=[],
        crew_analysis={},
        validated_results=None,
        
        # Final output
        final_report=None,
        
        # Audit and metadata
        processing_times={},
        data_quality_scores={},
        checkpoint_metadata={}
    )


def update_workflow_state(state: CompleteWorkflowState, 
                         updates: Dict[str, Any],
                         validate: bool = True) -> CompleteWorkflowState:
    """
    Update workflow state with validation.
    
    Args:
        state: Current workflow state
        updates: Dictionary of updates to apply
        validate: Whether to validate the state transition
        
    Returns:
        CompleteWorkflowState: Updated state
        
    Raises:
        ValueError: If validation fails
    """
    # Create new state with updates
    new_state = state.copy()
    new_state.update(updates)
    new_state["updated_at"] = datetime.now()
    
    if validate:
        # Validate state transition
        is_valid, errors = WorkflowStateValidator.validate_state_transition(state, new_state)
        if not is_valid:
            raise ValueError(f"Invalid state transition: {'; '.join(errors)}")
        
        # Validate data integrity
        is_valid, errors = WorkflowStateValidator.validate_data_integrity(new_state)
        if not is_valid:
            raise ValueError(f"Data integrity validation failed: {'; '.join(errors)}")
    
    return new_state