"""
Checkpoint Manager for LangGraph Workflow Orchestration

This module provides checkpoint-based persistence using SqliteSaver for
durable execution and recovery capabilities.
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .state_management import CompleteWorkflowState, WorkflowStateValidator


logger = logging.getLogger(__name__)


class WorkflowCheckpointManager:
    """
    Manages workflow checkpoints using SqliteSaver for persistent state storage.
    """
    
    def __init__(self, db_path: str = "checkpoints/workflow_checkpoints.db"):
        """
        Initialize the checkpoint manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._checkpointer: Optional[SqliteSaver] = None
        
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
    def get_checkpointer(self) -> SqliteSaver:
        """
        Get or create the SqliteSaver instance.
        
        Returns:
            SqliteSaver: Configured checkpointer instance
        """
        if self._checkpointer is None:
            # Use absolute path for SQLite connection
            abs_path = Path(self.db_path).absolute()
            self._checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{abs_path}")
            logger.info(f"Initialized SqliteSaver with database: {abs_path}")
        
        return self._checkpointer
    
    def save_checkpoint(self, workflow_id: str, state: CompleteWorkflowState, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a workflow checkpoint.
        
        Args:
            workflow_id: Unique workflow identifier
            state: Current workflow state
            metadata: Additional metadata to store
            
        Returns:
            str: Checkpoint ID
        """
        checkpointer = self.get_checkpointer()
        
        # Validate state before saving
        is_valid, errors = WorkflowStateValidator.validate_data_integrity(state)
        if not is_valid:
            logger.warning(f"State validation warnings for workflow {workflow_id}: {errors}")
        
        # Prepare checkpoint configuration
        config = {
            "configurable": {
                "thread_id": workflow_id,
                "checkpoint_ns": ""
            }
        }
        
        # Create checkpoint data
        checkpoint_id = f"checkpoint_{datetime.now().isoformat()}_{workflow_id}"
        checkpoint_data = {
            "v": 4,
            "ts": datetime.now().isoformat(),
            "id": checkpoint_id,
            "channel_values": {
                "workflow_state": state,
                "metadata": metadata or {}
            },
            "channel_versions": {
                "__start__": 1,
                "workflow_state": 1,
                "metadata": 1
            },
            "versions_seen": {
                "__input__": {},
                "__start__": {"__start__": 1}
            }
        }
        
        try:
            # Store checkpoint
            with checkpointer:
                checkpointer.put(config, checkpoint_data, {}, {})
            
            logger.info(f"Saved checkpoint {checkpoint_id} for workflow {workflow_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint for workflow {workflow_id}: {e}")
            raise
    
    def load_checkpoint(self, workflow_id: str, 
                       checkpoint_id: Optional[str] = None) -> Optional[CompleteWorkflowState]:
        """
        Load a workflow checkpoint.
        
        Args:
            workflow_id: Unique workflow identifier
            checkpoint_id: Specific checkpoint ID (if None, loads latest)
            
        Returns:
            CompleteWorkflowState or None: Loaded state or None if not found
        """
        checkpointer = self.get_checkpointer()
        
        config = {
            "configurable": {
                "thread_id": workflow_id
            }
        }
        
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        
        try:
            with checkpointer:
                checkpoint = checkpointer.get(config)
            
            if checkpoint and "channel_values" in checkpoint:
                workflow_state = checkpoint["channel_values"].get("workflow_state")
                if workflow_state:
                    logger.info(f"Loaded checkpoint for workflow {workflow_id}")
                    return workflow_state
            
            logger.warning(f"No checkpoint found for workflow {workflow_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint for workflow {workflow_id}: {e}")
            raise
    
    def list_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            List[Dict]: List of checkpoint metadata
        """
        checkpointer = self.get_checkpointer()
        
        config = {
            "configurable": {
                "thread_id": workflow_id
            }
        }
        
        try:
            with checkpointer:
                checkpoints = list(checkpointer.list(config))
            
            checkpoint_list = []
            for checkpoint in checkpoints:
                checkpoint_info = {
                    "id": checkpoint.get("id"),
                    "timestamp": checkpoint.get("ts"),
                    "step": checkpoint.get("channel_values", {}).get("workflow_state", {}).get("current_step"),
                    "has_errors": checkpoint.get("channel_values", {}).get("workflow_state", {}).get("has_errors", False)
                }
                checkpoint_list.append(checkpoint_info)
            
            logger.info(f"Found {len(checkpoint_list)} checkpoints for workflow {workflow_id}")
            return checkpoint_list
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints for workflow {workflow_id}: {e}")
            raise
    
    def delete_workflow_checkpoints(self, workflow_id: str) -> bool:
        """
        Delete all checkpoints for a workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            bool: True if successful
        """
        try:
            # Note: SqliteSaver doesn't have a direct delete method
            # This would require direct SQL operations
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete checkpoints for the specific thread_id
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", 
                (workflow_id,)
            )
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted {deleted_count} checkpoints for workflow {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoints for workflow {workflow_id}: {e}")
            return False
    
    def get_workflow_recovery_info(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get recovery information for a failed workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            Dict or None: Recovery information including last successful step
        """
        try:
            checkpoints = self.list_checkpoints(workflow_id)
            if not checkpoints:
                return None
            
            # Sort by timestamp to get the latest
            checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
            
            latest_checkpoint = checkpoints[0]
            state = self.load_checkpoint(workflow_id)
            
            if not state:
                return None
            
            recovery_info = {
                "workflow_id": workflow_id,
                "last_checkpoint_id": latest_checkpoint["id"],
                "last_successful_step": latest_checkpoint["step"],
                "has_errors": latest_checkpoint["has_errors"],
                "retry_count": state.get("retry_count", 0),
                "failed_steps": state.get("failed_steps", []),
                "error_message": state.get("error_message"),
                "can_recover": not state.get("has_errors", False) or state.get("retry_count", 0) < 3
            }
            
            return recovery_info
            
        except Exception as e:
            logger.error(f"Failed to get recovery info for workflow {workflow_id}: {e}")
            return None
    
    def cleanup_old_checkpoints(self, days_old: int = 30) -> int:
        """
        Clean up checkpoints older than specified days.
        
        Args:
            days_old: Number of days to keep checkpoints
            
        Returns:
            int: Number of checkpoints deleted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate cutoff date
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            # Delete old checkpoints
            cursor.execute(
                "DELETE FROM checkpoints WHERE created_at < ?", 
                (cutoff_date,)
            )
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0


class AsyncWorkflowCheckpointManager:
    """
    Asynchronous version of the checkpoint manager for async workflows.
    """
    
    def __init__(self, db_path: str = "checkpoints/async_workflow_checkpoints.db"):
        """
        Initialize the async checkpoint manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._checkpointer: Optional[AsyncSqliteSaver] = None
        
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    async def get_checkpointer(self) -> AsyncSqliteSaver:
        """
        Get or create the AsyncSqliteSaver instance.
        
        Returns:
            AsyncSqliteSaver: Configured async checkpointer instance
        """
        if self._checkpointer is None:
            self._checkpointer = AsyncSqliteSaver.from_conn_string(f"sqlite:///{self.db_path}")
            logger.info(f"Initialized AsyncSqliteSaver with database: {self.db_path}")
        
        return self._checkpointer
    
    async def save_checkpoint(self, workflow_id: str, state: CompleteWorkflowState, 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Asynchronously save a workflow checkpoint.
        
        Args:
            workflow_id: Unique workflow identifier
            state: Current workflow state
            metadata: Additional metadata to store
            
        Returns:
            str: Checkpoint ID
        """
        checkpointer = await self.get_checkpointer()
        
        # Validate state before saving
        is_valid, errors = WorkflowStateValidator.validate_data_integrity(state)
        if not is_valid:
            logger.warning(f"State validation warnings for workflow {workflow_id}: {errors}")
        
        # Prepare checkpoint configuration
        config = {
            "configurable": {
                "thread_id": workflow_id,
                "checkpoint_ns": ""
            }
        }
        
        # Create checkpoint data
        checkpoint_id = f"checkpoint_{datetime.now().isoformat()}_{workflow_id}"
        checkpoint_data = {
            "v": 4,
            "ts": datetime.now().isoformat(),
            "id": checkpoint_id,
            "channel_values": {
                "workflow_state": state,
                "metadata": metadata or {}
            },
            "channel_versions": {
                "__start__": 1,
                "workflow_state": 1,
                "metadata": 1
            },
            "versions_seen": {
                "__input__": {},
                "__start__": {"__start__": 1}
            }
        }
        
        try:
            # Store checkpoint asynchronously
            async with checkpointer:
                await checkpointer.aput(config, checkpoint_data, {}, {})
            
            logger.info(f"Saved async checkpoint {checkpoint_id} for workflow {workflow_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save async checkpoint for workflow {workflow_id}: {e}")
            raise
    
    async def load_checkpoint(self, workflow_id: str, 
                            checkpoint_id: Optional[str] = None) -> Optional[CompleteWorkflowState]:
        """
        Asynchronously load a workflow checkpoint.
        
        Args:
            workflow_id: Unique workflow identifier
            checkpoint_id: Specific checkpoint ID (if None, loads latest)
            
        Returns:
            CompleteWorkflowState or None: Loaded state or None if not found
        """
        checkpointer = await self.get_checkpointer()
        
        config = {
            "configurable": {
                "thread_id": workflow_id
            }
        }
        
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        
        try:
            async with checkpointer:
                checkpoint = await checkpointer.aget(config)
            
            if checkpoint and "channel_values" in checkpoint:
                workflow_state = checkpoint["channel_values"].get("workflow_state")
                if workflow_state:
                    logger.info(f"Loaded async checkpoint for workflow {workflow_id}")
                    return workflow_state
            
            logger.warning(f"No async checkpoint found for workflow {workflow_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load async checkpoint for workflow {workflow_id}: {e}")
            raise