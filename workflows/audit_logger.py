"""
Audit Logger for Workflow Metadata Tracking

This module provides comprehensive audit logging and metadata tracking
for workflow execution, state transitions, and performance monitoring.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from .state_management import CompleteWorkflowState


class AuditEventType(Enum):
    """Types of audit events."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    STATE_TRANSITION = "state_transition"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"
    ERROR_OCCURRED = "error_occurred"
    RETRY_ATTEMPTED = "retry_attempted"
    DATA_COLLECTED = "data_collected"
    MODEL_TRAINED = "model_trained"
    PREDICTION_GENERATED = "prediction_generated"
    ANALYSIS_COMPLETED = "analysis_completed"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class AuditEvent:
    """Represents a single audit event."""
    event_id: str
    workflow_id: str
    event_type: AuditEventType
    timestamp: datetime
    step: str
    message: str
    metadata: Dict[str, Any]
    duration_ms: Optional[float] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type
        data['timestamp'] = self.timestamp.isoformat()
        return data


class WorkflowAuditLogger:
    """
    Comprehensive audit logger for workflow execution tracking.
    """
    
    def __init__(self, log_dir: str = "logs/audit", 
                 log_level: int = logging.INFO):
        """
        Initialize the audit logger.
        
        Args:
            log_dir: Directory for audit log files
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up structured logging
        self.logger = logging.getLogger("workflow_audit")
        self.logger.setLevel(log_level)
        
        # Create file handler for audit logs
        audit_log_file = self.log_dir / "workflow_audit.log"
        file_handler = logging.FileHandler(audit_log_file)
        file_handler.setLevel(log_level)
        
        # Create JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        
        # In-memory event storage for analysis
        self._events: List[AuditEvent] = []
        
    def log_event(self, event: AuditEvent):
        """
        Log an audit event.
        
        Args:
            event: AuditEvent to log
        """
        # Add to in-memory storage
        self._events.append(event)
        
        # Log to file
        event_data = event.to_dict()
        self.logger.info(json.dumps(event_data))
        
        # Also log to console for important events
        if event.event_type in [
            AuditEventType.WORKFLOW_STARTED,
            AuditEventType.WORKFLOW_COMPLETED,
            AuditEventType.WORKFLOW_FAILED,
            AuditEventType.ERROR_OCCURRED
        ]:
            console_logger = logging.getLogger("workflow_console")
            console_logger.info(f"[{event.workflow_id}] {event.message}")
    
    def log_workflow_started(self, workflow_id: str, suppliers_count: int, 
                           config: Dict[str, Any]):
        """Log workflow start event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_start_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.WORKFLOW_STARTED,
            timestamp=datetime.now(),
            step="initialize_workflow",
            message=f"Workflow started with {suppliers_count} suppliers",
            metadata={
                "suppliers_count": suppliers_count,
                "config": config
            }
        )
        self.log_event(event)
    
    def log_workflow_completed(self, workflow_id: str, total_duration_ms: float,
                             final_report: Dict[str, Any]):
        """Log workflow completion event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_complete_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.WORKFLOW_COMPLETED,
            timestamp=datetime.now(),
            step="completed",
            message=f"Workflow completed successfully in {total_duration_ms:.2f}ms",
            metadata={
                "total_duration_ms": total_duration_ms,
                "final_report_summary": {
                    "suppliers_analyzed": len(final_report.get("detailed_results", {}).get("suppliers", [])),
                    "risk_events_generated": len(final_report.get("detailed_results", {}).get("risk_events", [])),
                    "overall_risk_score": final_report.get("summary", {}).get("overall_risk_score")
                }
            },
            duration_ms=total_duration_ms
        )
        self.log_event(event)
    
    def log_workflow_failed(self, workflow_id: str, step: str, error_message: str,
                          total_duration_ms: float, retry_count: int):
        """Log workflow failure event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_failed_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.WORKFLOW_FAILED,
            timestamp=datetime.now(),
            step=step,
            message=f"Workflow failed at step '{step}' after {retry_count} retries",
            metadata={
                "retry_count": retry_count,
                "total_duration_ms": total_duration_ms
            },
            duration_ms=total_duration_ms,
            error_details=error_message
        )
        self.log_event(event)
    
    def log_state_transition(self, workflow_id: str, from_step: str, to_step: str,
                           duration_ms: float, metadata: Optional[Dict[str, Any]] = None):
        """Log state transition event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_transition_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.STATE_TRANSITION,
            timestamp=datetime.now(),
            step=to_step,
            message=f"State transition: {from_step} -> {to_step}",
            metadata={
                "from_step": from_step,
                "to_step": to_step,
                "step_metadata": metadata or {}
            },
            duration_ms=duration_ms
        )
        self.log_event(event)
    
    def log_checkpoint_saved(self, workflow_id: str, checkpoint_id: str, step: str):
        """Log checkpoint save event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_checkpoint_save_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.CHECKPOINT_SAVED,
            timestamp=datetime.now(),
            step=step,
            message=f"Checkpoint saved: {checkpoint_id}",
            metadata={
                "checkpoint_id": checkpoint_id
            }
        )
        self.log_event(event)
    
    def log_checkpoint_loaded(self, workflow_id: str, checkpoint_id: str, step: str):
        """Log checkpoint load event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_checkpoint_load_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.CHECKPOINT_LOADED,
            timestamp=datetime.now(),
            step=step,
            message=f"Checkpoint loaded: {checkpoint_id}",
            metadata={
                "checkpoint_id": checkpoint_id
            }
        )
        self.log_event(event)
    
    def log_error(self, workflow_id: str, step: str, error_message: str,
                 error_type: str, retry_count: int):
        """Log error event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_error_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.ERROR_OCCURRED,
            timestamp=datetime.now(),
            step=step,
            message=f"Error in step '{step}': {error_type}",
            metadata={
                "error_type": error_type,
                "retry_count": retry_count
            },
            error_details=error_message
        )
        self.log_event(event)
    
    def log_retry_attempt(self, workflow_id: str, step: str, retry_count: int,
                         max_retries: int):
        """Log retry attempt event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_retry_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.RETRY_ATTEMPTED,
            timestamp=datetime.now(),
            step=step,
            message=f"Retry attempt {retry_count}/{max_retries} for step '{step}'",
            metadata={
                "retry_count": retry_count,
                "max_retries": max_retries
            }
        )
        self.log_event(event)
    
    def log_data_collection(self, workflow_id: str, data_type: str, 
                          records_collected: int, quality_score: float,
                          duration_ms: float):
        """Log data collection event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_data_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.DATA_COLLECTED,
            timestamp=datetime.now(),
            step="collect_data",
            message=f"Collected {records_collected} {data_type} records",
            metadata={
                "data_type": data_type,
                "records_collected": records_collected,
                "quality_score": quality_score
            },
            duration_ms=duration_ms
        )
        self.log_event(event)
    
    def log_model_training(self, workflow_id: str, model_type: str,
                         performance_metrics: Dict[str, float], duration_ms: float):
        """Log model training event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_training_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.MODEL_TRAINED,
            timestamp=datetime.now(),
            step="train_ml_models",
            message=f"Trained {model_type} model",
            metadata={
                "model_type": model_type,
                "performance_metrics": performance_metrics
            },
            duration_ms=duration_ms
        )
        self.log_event(event)
    
    def log_predictions_generated(self, workflow_id: str, predictions_count: int,
                                average_confidence: float, duration_ms: float):
        """Log prediction generation event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_predictions_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.PREDICTION_GENERATED,
            timestamp=datetime.now(),
            step="generate_ml_predictions",
            message=f"Generated {predictions_count} predictions",
            metadata={
                "predictions_count": predictions_count,
                "average_confidence": average_confidence
            },
            duration_ms=duration_ms
        )
        self.log_event(event)
    
    def log_analysis_completed(self, workflow_id: str, analysis_type: str,
                             insights_count: int, duration_ms: float):
        """Log analysis completion event."""
        event = AuditEvent(
            event_id=f"{workflow_id}_analysis_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.ANALYSIS_COMPLETED,
            timestamp=datetime.now(),
            step="run_crew_analysis",
            message=f"Completed {analysis_type} analysis with {insights_count} insights",
            metadata={
                "analysis_type": analysis_type,
                "insights_count": insights_count
            },
            duration_ms=duration_ms
        )
        self.log_event(event)
    
    def log_performance_metric(self, workflow_id: str, metric_name: str,
                             metric_value: float, step: str):
        """Log performance metric."""
        event = AuditEvent(
            event_id=f"{workflow_id}_metric_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            event_type=AuditEventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            step=step,
            message=f"Performance metric: {metric_name} = {metric_value}",
            metadata={
                "metric_name": metric_name,
                "metric_value": metric_value
            }
        )
        self.log_event(event)
    
    def get_workflow_events(self, workflow_id: str) -> List[AuditEvent]:
        """Get all events for a specific workflow."""
        return [event for event in self._events if event.workflow_id == workflow_id]
    
    def get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get summary statistics for a workflow."""
        events = self.get_workflow_events(workflow_id)
        
        if not events:
            return {"error": "No events found for workflow"}
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        start_event = next((e for e in events if e.event_type == AuditEventType.WORKFLOW_STARTED), None)
        end_event = next((e for e in events if e.event_type in [
            AuditEventType.WORKFLOW_COMPLETED, AuditEventType.WORKFLOW_FAILED
        ]), None)
        
        total_duration = None
        if start_event and end_event:
            total_duration = (end_event.timestamp - start_event.timestamp).total_seconds() * 1000
        
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Get error events
        error_events = [e for e in events if e.event_type == AuditEventType.ERROR_OCCURRED]
        
        # Calculate step durations
        step_durations = {}
        for event in events:
            if event.duration_ms:
                step_durations[event.step] = step_durations.get(event.step, 0) + event.duration_ms
        
        return {
            "workflow_id": workflow_id,
            "total_events": len(events),
            "total_duration_ms": total_duration,
            "event_counts": event_counts,
            "error_count": len(error_events),
            "step_durations": step_durations,
            "start_time": start_event.timestamp.isoformat() if start_event else None,
            "end_time": end_event.timestamp.isoformat() if end_event else None,
            "status": "completed" if any(e.event_type == AuditEventType.WORKFLOW_COMPLETED for e in events) else "failed" if error_events else "running"
        }
    
    def export_workflow_audit(self, workflow_id: str, output_file: Optional[str] = None) -> str:
        """Export workflow audit data to JSON file."""
        events = self.get_workflow_events(workflow_id)
        summary = self.get_workflow_summary(workflow_id)
        
        audit_data = {
            "workflow_id": workflow_id,
            "summary": summary,
            "events": [event.to_dict() for event in events]
        }
        
        if not output_file:
            output_file = self.log_dir / f"workflow_{workflow_id}_audit.json"
        
        with open(output_file, 'w') as f:
            json.dump(audit_data, f, indent=2, default=str)
        
        return str(output_file)