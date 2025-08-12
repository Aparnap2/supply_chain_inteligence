"""
Risk event tracking and history management system.
"""

import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict

from .risk_event import RiskEvent, RiskLevel
from .supplier import Supplier

logger = logging.getLogger(__name__)


class RiskEventTracker:
    """
    Manages risk event storage, tracking, and historical analysis.
    """
    
    def __init__(self, db_path: str = "risk_events.db"):
        """
        Initialize the risk event tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.correlation_threshold = 0.7
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Risk events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_events (
                        event_id TEXT PRIMARY KEY,
                        supplier_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        probability REAL NOT NULL,
                        impact_score REAL NOT NULL,
                        confidence_level REAL NOT NULL,
                        description TEXT NOT NULL,
                        evidence TEXT,  -- JSON array
                        data_sources TEXT,  -- JSON array
                        geographic_scope TEXT,
                        predicted_timeline TEXT,
                        mitigation_actions TEXT,  -- JSON array
                        status TEXT NOT NULL DEFAULT 'active',
                        detected_at TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        resolved_at TEXT,
                        resolution_notes TEXT
                    )
                ''')
                
                # Risk event history table for tracking changes
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_event_history (
                        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        change_type TEXT NOT NULL,  -- created, updated, status_changed, resolved
                        old_values TEXT,  -- JSON
                        new_values TEXT,  -- JSON
                        changed_by TEXT,
                        change_reason TEXT,
                        changed_at TEXT NOT NULL,
                        FOREIGN KEY (event_id) REFERENCES risk_events (event_id)
                    )
                ''')
                
                # Risk event correlations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_event_correlations (
                        correlation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id_1 TEXT NOT NULL,
                        event_id_2 TEXT NOT NULL,
                        correlation_score REAL NOT NULL,
                        correlation_type TEXT NOT NULL,  -- temporal, geographic, supplier, industry
                        detected_at TEXT NOT NULL,
                        FOREIGN KEY (event_id_1) REFERENCES risk_events (event_id),
                        FOREIGN KEY (event_id_2) REFERENCES risk_events (event_id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_supplier_id ON risk_events (supplier_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON risk_events (event_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON risk_events (severity)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON risk_events (status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_detected_at ON risk_events (detected_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_event_id ON risk_event_history (event_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_correlations_events ON risk_event_correlations (event_id_1, event_id_2)')
                
                conn.commit()
                logger.info("Risk event database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_risk_event(self, risk_event: RiskEvent, changed_by: str = "system") -> bool:
        """
        Store a new risk event in the database.
        
        Args:
            risk_event: RiskEvent object to store
            changed_by: Who created the event
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if event already exists
                cursor.execute('SELECT event_id FROM risk_events WHERE event_id = ?', (risk_event.event_id,))
                if cursor.fetchone():
                    logger.warning(f"Risk event {risk_event.event_id} already exists")
                    return False
                
                # Insert risk event
                cursor.execute('''
                    INSERT INTO risk_events (
                        event_id, supplier_id, event_type, severity, probability,
                        impact_score, confidence_level, description, evidence,
                        data_sources, geographic_scope, predicted_timeline,
                        mitigation_actions, status, detected_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    risk_event.event_id,
                    risk_event.supplier_id,
                    risk_event.event_type,
                    risk_event.severity if isinstance(risk_event.severity, str) else risk_event.severity.value,
                    risk_event.probability,
                    risk_event.impact_score,
                    risk_event.confidence_level,
                    risk_event.description,
                    json.dumps(risk_event.evidence),
                    json.dumps([ds.value for ds in risk_event.data_sources]),
                    risk_event.geographic_scope,
                    risk_event.predicted_timeline,
                    json.dumps(risk_event.mitigation_actions),
                    risk_event.status,
                    risk_event.detected_at.isoformat(),
                    risk_event.created_at.isoformat(),
                    risk_event.updated_at.isoformat()
                ))
                
                # Record creation in history
                self._record_history_change(
                    cursor, risk_event.event_id, "created", {}, 
                    risk_event.dict(), changed_by, "Initial risk event creation"
                )
                
                conn.commit()
                logger.info(f"Stored risk event {risk_event.event_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store risk event {risk_event.event_id}: {e}")
            return False
    
    def update_risk_event(
        self, 
        event_id: str, 
        updates: Dict[str, Any], 
        changed_by: str = "system",
        change_reason: str = "Event update"
    ) -> bool:
        """
        Update an existing risk event.
        
        Args:
            event_id: ID of the event to update
            updates: Dictionary of fields to update
            changed_by: Who made the update
            change_reason: Reason for the update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current values
                cursor.execute('SELECT * FROM risk_events WHERE event_id = ?', (event_id,))
                current_row = cursor.fetchone()
                if not current_row:
                    logger.warning(f"Risk event {event_id} not found")
                    return False
                
                # Convert row to dict
                columns = [desc[0] for desc in cursor.description]
                current_values = dict(zip(columns, current_row))
                
                # Prepare update query
                update_fields = []
                update_values = []
                
                for field, value in updates.items():
                    if field in ['evidence', 'data_sources', 'mitigation_actions']:
                        # JSON fields
                        update_fields.append(f"{field} = ?")
                        update_values.append(json.dumps(value))
                    elif field == 'severity':
                        # Enum field - handle both enum and string values
                        update_fields.append(f"{field} = ?")
                        update_values.append(value if isinstance(value, str) else value.value)
                    elif field in columns:
                        update_fields.append(f"{field} = ?")
                        update_values.append(value)
                
                # Always update the updated_at timestamp
                update_fields.append("updated_at = ?")
                update_values.append(datetime.now().isoformat())
                
                if not update_fields:
                    logger.warning("No valid fields to update")
                    return False
                
                # Execute update
                update_query = f"UPDATE risk_events SET {', '.join(update_fields)} WHERE event_id = ?"
                update_values.append(event_id)
                cursor.execute(update_query, update_values)
                
                # Record change in history
                self._record_history_change(
                    cursor, event_id, "updated", current_values, 
                    updates, changed_by, change_reason
                )
                
                conn.commit()
                logger.info(f"Updated risk event {event_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update risk event {event_id}: {e}")
            return False
    
    def change_event_status(
        self, 
        event_id: str, 
        new_status: str, 
        changed_by: str = "system",
        resolution_notes: str = None
    ) -> bool:
        """
        Change the status of a risk event.
        
        Args:
            event_id: ID of the event
            new_status: New status (active, monitoring, resolved, false_positive)
            changed_by: Who changed the status
            resolution_notes: Optional notes for resolution
            
        Returns:
            True if successful, False otherwise
        """
        valid_statuses = ['active', 'monitoring', 'resolved', 'false_positive']
        if new_status not in valid_statuses:
            logger.error(f"Invalid status: {new_status}. Must be one of {valid_statuses}")
            return False
        
        updates = {'status': new_status}
        
        # Add resolution timestamp and notes if resolving
        if new_status in ['resolved', 'false_positive']:
            updates['resolved_at'] = datetime.now().isoformat()
            if resolution_notes:
                updates['resolution_notes'] = resolution_notes
        
        return self.update_risk_event(
            event_id, updates, changed_by, f"Status changed to {new_status}"
        )
    
    def get_risk_event(self, event_id: str) -> Optional[RiskEvent]:
        """
        Retrieve a risk event by ID.
        
        Args:
            event_id: ID of the event to retrieve
            
        Returns:
            RiskEvent object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM risk_events WHERE event_id = ?', (event_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_risk_event(cursor, row)
                
        except Exception as e:
            logger.error(f"Failed to retrieve risk event {event_id}: {e}")
            return None
    
    def get_supplier_risk_events(
        self, 
        supplier_id: str, 
        status_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[RiskEvent]:
        """
        Get all risk events for a specific supplier.
        
        Args:
            supplier_id: ID of the supplier
            status_filter: Optional status filter
            limit: Optional limit on number of results
            
        Returns:
            List of RiskEvent objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM risk_events WHERE supplier_id = ?'
                params = [supplier_id]
                
                if status_filter:
                    query += ' AND status = ?'
                    params.append(status_filter)
                
                query += ' ORDER BY detected_at DESC'
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_risk_event(cursor, row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to retrieve risk events for supplier {supplier_id}: {e}")
            return []
    
    def get_risk_events_by_criteria(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[RiskEvent]:
        """
        Get risk events based on various criteria.
        
        Args:
            event_type: Filter by event type
            severity: Filter by severity level
            status: Filter by status
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Limit number of results
            
        Returns:
            List of RiskEvent objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM risk_events WHERE 1=1'
                params = []
                
                if event_type:
                    query += ' AND event_type = ?'
                    params.append(event_type)
                
                if severity:
                    query += ' AND severity = ?'
                    params.append(severity)
                
                if status:
                    query += ' AND status = ?'
                    params.append(status)
                
                if start_date:
                    query += ' AND detected_at >= ?'
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += ' AND detected_at <= ?'
                    params.append(end_date.isoformat())
                
                query += ' ORDER BY detected_at DESC'
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_risk_event(cursor, row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to retrieve risk events by criteria: {e}")
            return []
    
    def get_event_history(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get the change history for a risk event.
        
        Args:
            event_id: ID of the event
            
        Returns:
            List of history records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM risk_event_history 
                    WHERE event_id = ? 
                    ORDER BY changed_at DESC
                ''', (event_id,))
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    record = dict(zip(columns, row))
                    # Parse JSON fields
                    if record['old_values']:
                        record['old_values'] = json.loads(record['old_values'])
                    if record['new_values']:
                        record['new_values'] = json.loads(record['new_values'])
                    history.append(record)
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to retrieve history for event {event_id}: {e}")
            return []
    
    def analyze_risk_trends(
        self, 
        days: int = 30,
        supplier_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze risk event trends over time.
        
        Args:
            days: Number of days to analyze
            supplier_id: Optional supplier filter
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Base query
                base_query = '''
                    SELECT event_type, severity, detected_at, supplier_id
                    FROM risk_events 
                    WHERE detected_at >= ?
                '''
                params = [start_date.isoformat()]
                
                if supplier_id:
                    base_query += ' AND supplier_id = ?'
                    params.append(supplier_id)
                
                cursor.execute(base_query, params)
                rows = cursor.fetchall()
                
                # Analyze trends
                trends = {
                    'total_events': len(rows),
                    'events_by_type': defaultdict(int),
                    'events_by_severity': defaultdict(int),
                    'events_by_day': defaultdict(int),
                    'unique_suppliers': set(),
                    'trend_direction': 'stable'
                }
                
                for row in rows:
                    event_type, severity, detected_at, supplier_id = row
                    
                    trends['events_by_type'][event_type] += 1
                    trends['events_by_severity'][severity] += 1
                    trends['unique_suppliers'].add(supplier_id)
                    
                    # Group by day
                    day = detected_at[:10]  # YYYY-MM-DD
                    trends['events_by_day'][day] += 1
                
                # Convert sets to counts
                trends['unique_suppliers'] = len(trends['unique_suppliers'])
                
                # Calculate trend direction
                if len(trends['events_by_day']) > 1:
                    daily_counts = list(trends['events_by_day'].values())
                    recent_avg = sum(daily_counts[-7:]) / min(7, len(daily_counts))
                    earlier_avg = sum(daily_counts[:-7]) / max(1, len(daily_counts) - 7)
                    
                    if recent_avg > earlier_avg * 1.2:
                        trends['trend_direction'] = 'increasing'
                    elif recent_avg < earlier_avg * 0.8:
                        trends['trend_direction'] = 'decreasing'
                
                # Convert defaultdicts to regular dicts
                trends['events_by_type'] = dict(trends['events_by_type'])
                trends['events_by_severity'] = dict(trends['events_by_severity'])
                trends['events_by_day'] = dict(trends['events_by_day'])
                
                return trends
                
        except Exception as e:
            logger.error(f"Failed to analyze risk trends: {e}")
            return {}
    
    def detect_correlations(self, min_correlation_score: float = 0.7) -> List[Dict[str, Any]]:
        """
        Detect correlations between risk events.
        
        Args:
            min_correlation_score: Minimum correlation score to report
            
        Returns:
            List of correlation records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent active events
                cursor.execute('''
                    SELECT event_id, supplier_id, event_type, severity, 
                           detected_at, geographic_scope
                    FROM risk_events 
                    WHERE status = 'active' 
                    AND detected_at >= ?
                    ORDER BY detected_at DESC
                ''', [(datetime.now() - timedelta(days=90)).isoformat()])
                
                events = cursor.fetchall()
                correlations = []
                
                # Compare each pair of events
                for i, event1 in enumerate(events):
                    for event2 in events[i+1:]:
                        correlation = self._calculate_correlation(event1, event2)
                        
                        if correlation['score'] >= min_correlation_score:
                            # Store correlation if not already exists
                            cursor.execute('''
                                SELECT correlation_id FROM risk_event_correlations
                                WHERE (event_id_1 = ? AND event_id_2 = ?)
                                OR (event_id_1 = ? AND event_id_2 = ?)
                            ''', (event1[0], event2[0], event2[0], event1[0]))
                            
                            if not cursor.fetchone():
                                cursor.execute('''
                                    INSERT INTO risk_event_correlations
                                    (event_id_1, event_id_2, correlation_score, 
                                     correlation_type, detected_at)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', (
                                    event1[0], event2[0], correlation['score'],
                                    correlation['type'], datetime.now().isoformat()
                                ))
                            
                            correlations.append({
                                'event_1': event1[0],
                                'event_2': event2[0],
                                'correlation_score': correlation['score'],
                                'correlation_type': correlation['type'],
                                'description': correlation['description']
                            })
                
                conn.commit()
                logger.info(f"Detected {len(correlations)} correlations")
                return correlations
                
        except Exception as e:
            logger.error(f"Failed to detect correlations: {e}")
            return []
    
    def _calculate_correlation(self, event1: Tuple, event2: Tuple) -> Dict[str, Any]:
        """Calculate correlation between two events."""
        event1_id, event1_supplier, event1_type, event1_severity, event1_time, event1_geo = event1
        event2_id, event2_supplier, event2_type, event2_severity, event2_time, event2_geo = event2
        
        correlation_score = 0.0
        correlation_type = "unknown"
        description = ""
        
        # Temporal correlation (events close in time)
        time1 = datetime.fromisoformat(event1_time)
        time2 = datetime.fromisoformat(event2_time)
        time_diff = abs((time1 - time2).total_seconds() / 3600)  # Hours
        
        if time_diff <= 24:  # Within 24 hours
            correlation_score += 0.4
            correlation_type = "temporal"
            description = f"Events occurred within {time_diff:.1f} hours"
        elif time_diff <= 168:  # Within 1 week
            correlation_score += 0.2
        
        # Geographic correlation
        if event1_geo and event2_geo and event1_geo == event2_geo:
            correlation_score += 0.3
            if correlation_type == "temporal":
                correlation_type = "temporal_geographic"
            else:
                correlation_type = "geographic"
            description += f" in same geographic area ({event1_geo})"
        
        # Supplier correlation
        if event1_supplier == event2_supplier:
            correlation_score += 0.5
            correlation_type = "supplier"
            description = f"Same supplier affected: {event1_supplier}"
        
        # Event type correlation
        if event1_type == event2_type:
            correlation_score += 0.2
            description += f" (same event type: {event1_type})"
        
        # Severity correlation
        if event1_severity == event2_severity and event1_severity in ['high', 'critical']:
            correlation_score += 0.1
        
        return {
            'score': min(1.0, correlation_score),
            'type': correlation_type,
            'description': description.strip()
        }
    
    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get statistics about risk event lifecycle management."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Overall statistics
                cursor.execute('SELECT COUNT(*) FROM risk_events')
                total_events = cursor.fetchone()[0]
                
                cursor.execute('SELECT status, COUNT(*) FROM risk_events GROUP BY status')
                status_counts = dict(cursor.fetchall())
                
                cursor.execute('SELECT severity, COUNT(*) FROM risk_events GROUP BY severity')
                severity_counts = dict(cursor.fetchall())
                
                # Resolution time analysis
                cursor.execute('''
                    SELECT AVG(
                        (julianday(resolved_at) - julianday(detected_at)) * 24
                    ) as avg_resolution_hours
                    FROM risk_events 
                    WHERE resolved_at IS NOT NULL
                ''')
                avg_resolution_hours = cursor.fetchone()[0] or 0
                
                # Recent activity
                cursor.execute('''
                    SELECT COUNT(*) FROM risk_events 
                    WHERE detected_at >= ?
                ''', [(datetime.now() - timedelta(days=7)).isoformat()])
                recent_events = cursor.fetchone()[0]
                
                return {
                    'total_events': total_events,
                    'status_distribution': status_counts,
                    'severity_distribution': severity_counts,
                    'average_resolution_hours': round(avg_resolution_hours, 2),
                    'recent_events_7_days': recent_events,
                    'resolution_rate': (
                        status_counts.get('resolved', 0) / max(1, total_events) * 100
                    )
                }
                
        except Exception as e:
            logger.error(f"Failed to get lifecycle statistics: {e}")
            return {}
    
    def _record_history_change(
        self, 
        cursor, 
        event_id: str, 
        change_type: str, 
        old_values: Dict, 
        new_values: Dict,
        changed_by: str, 
        change_reason: str
    ):
        """Record a change in the event history table."""
        cursor.execute('''
            INSERT INTO risk_event_history
            (event_id, change_type, old_values, new_values, 
             changed_by, change_reason, changed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, change_type, 
            json.dumps(old_values), json.dumps(new_values),
            changed_by, change_reason, datetime.now().isoformat()
        ))
    
    def _row_to_risk_event(self, cursor, row) -> RiskEvent:
        """Convert database row to RiskEvent object."""
        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, row))
        
        # Parse JSON fields
        data['evidence'] = json.loads(data['evidence']) if data['evidence'] else []
        data['data_sources'] = [
            getattr(__import__('models.risk_event', fromlist=['DataSource']), 'DataSource')(ds) 
            for ds in json.loads(data['data_sources'])
        ] if data['data_sources'] else []
        data['mitigation_actions'] = json.loads(data['mitigation_actions']) if data['mitigation_actions'] else []
        
        # Parse datetime fields
        data['detected_at'] = datetime.fromisoformat(data['detected_at'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        if data['resolved_at']:
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        
        # Convert severity to enum
        data['severity'] = RiskLevel(data['severity'])
        
        # Remove database-specific fields
        db_fields = ['resolved_at', 'resolution_notes']
        for field in db_fields:
            data.pop(field, None)
        
        return RiskEvent(**data)
    
    def cleanup_old_events(self, days_to_keep: int = 365) -> int:
        """
        Clean up old resolved events.
        
        Args:
            days_to_keep: Number of days to keep resolved events
            
        Returns:
            Number of events cleaned up
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count events to be deleted
                cursor.execute('''
                    SELECT COUNT(*) FROM risk_events 
                    WHERE status IN ('resolved', 'false_positive')
                    AND resolved_at < ?
                ''', [cutoff_date.isoformat()])
                
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Delete old events and their history
                    cursor.execute('''
                        DELETE FROM risk_event_history 
                        WHERE event_id IN (
                            SELECT event_id FROM risk_events 
                            WHERE status IN ('resolved', 'false_positive')
                            AND resolved_at < ?
                        )
                    ''', [cutoff_date.isoformat()])
                    
                    cursor.execute('''
                        DELETE FROM risk_events 
                        WHERE status IN ('resolved', 'false_positive')
                        AND resolved_at < ?
                    ''', [cutoff_date.isoformat()])
                    
                    conn.commit()
                    logger.info(f"Cleaned up {count} old risk events")
                
                return count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")
            return 0
    
    def export_events_to_json(self, filepath: str, criteria: Dict[str, Any] = None) -> bool:
        """
        Export risk events to JSON file.
        
        Args:
            filepath: Path to export file
            criteria: Optional criteria for filtering events
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get events based on criteria
            events = self.get_risk_events_by_criteria(**(criteria or {}))
            
            # Convert to serializable format
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_events': len(events),
                'events': [event.dict() for event in events]
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(events)} events to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export events: {e}")
            return False