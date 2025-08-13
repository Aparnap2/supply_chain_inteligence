"""
Risk event alerting and notification system for Streamlit dashboard.
"""

import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque

from .risk_event import RiskEvent, RiskLevel
from .supplier import Supplier

logger = logging.getLogger(__name__)


class AlertPriority(str, Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status values."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class RiskAlert:
    """
    Represents a risk alert with notification details.
    """
    
    def __init__(
        self,
        alert_id: str,
        event_id: str,
        supplier_id: str,
        alert_type: str,
        priority: AlertPriority,
        title: str,
        message: str,
        threshold_breached: str,
        recommended_actions: List[str] = None,
        escalation_level: int = 0
    ):
        self.alert_id = alert_id
        self.event_id = event_id
        self.supplier_id = supplier_id
        self.alert_type = alert_type
        self.priority = priority
        self.title = title
        self.message = message
        self.threshold_breached = threshold_breached
        self.recommended_actions = recommended_actions or []
        self.escalation_level = escalation_level
        self.status = AlertStatus.ACTIVE
        self.created_at = datetime.now()
        self.acknowledged_at = None
        self.acknowledged_by = None
        self.resolved_at = None
        self.dismissed_at = None
        self.escalated_at = None
        self.notification_count = 0
        self.last_notification = None
    
    def acknowledge(self, acknowledged_by: str) -> bool:
        """Acknowledge the alert."""
        if self.status == AlertStatus.ACTIVE:
            self.status = AlertStatus.ACKNOWLEDGED
            self.acknowledged_at = datetime.now()
            self.acknowledged_by = acknowledged_by
            return True
        return False
    
    def resolve(self) -> bool:
        """Resolve the alert."""
        if self.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
            self.status = AlertStatus.RESOLVED
            self.resolved_at = datetime.now()
            return True
        return False
    
    def dismiss(self) -> bool:
        """Dismiss the alert."""
        if self.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
            self.status = AlertStatus.DISMISSED
            self.dismissed_at = datetime.now()
            return True
        return False
    
    def escalate(self) -> bool:
        """Escalate the alert to next level."""
        if self.escalation_level < 3:  # Max 3 escalation levels
            self.escalation_level += 1
            self.escalated_at = datetime.now()
            return True
        return False
    
    def record_notification(self):
        """Record that a notification was sent."""
        self.notification_count += 1
        self.last_notification = datetime.now()
    
    def is_overdue(self, max_age_hours: int = 24) -> bool:
        """Check if alert is overdue for acknowledgment."""
        if self.status != AlertStatus.ACTIVE:
            return False
        
        age_hours = (datetime.now() - self.created_at).total_seconds() / 3600
        return age_hours > max_age_hours
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'event_id': self.event_id,
            'supplier_id': self.supplier_id,
            'alert_type': self.alert_type,
            'priority': self.priority,
            'title': self.title,
            'message': self.message,
            'threshold_breached': self.threshold_breached,
            'recommended_actions': self.recommended_actions,
            'escalation_level': self.escalation_level,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'dismissed_at': self.dismissed_at.isoformat() if self.dismissed_at else None,
            'escalated_at': self.escalated_at.isoformat() if self.escalated_at else None,
            'notification_count': self.notification_count,
            'last_notification': self.last_notification.isoformat() if self.last_notification else None
        }


class RiskAlertSystem:
    """
    Manages risk event alerts and notifications for the Streamlit dashboard.
    """
    
    def __init__(self):
        """Initialize the alert system."""
        self.alerts: Dict[str, RiskAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self.notification_callbacks: List[Callable] = []
        self.alert_rules = self._initialize_alert_rules()
        self.escalation_rules = self._initialize_escalation_rules()
        self.dashboard_notifications: deque = deque(maxlen=50)  # For Streamlit display
        
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threshold-based alert rules."""
        return {
            'high_risk_supplier': {
                'condition': lambda event: event.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL],
                'priority': AlertPriority.HIGH,
                'title_template': 'High Risk Supplier Alert: {supplier_name}',
                'message_template': 'Supplier {supplier_name} has been flagged with {severity} risk level.',
                'threshold': 'Risk Level >= High'
            },
            'critical_probability': {
                'condition': lambda event: event.probability >= 0.8,
                'priority': AlertPriority.CRITICAL,
                'title_template': 'Critical Probability Alert: {supplier_name}',
                'message_template': 'High probability ({probability:.1%}) of disruption for {supplier_name}.',
                'threshold': 'Probability >= 80%'
            },
            'high_impact_score': {
                'condition': lambda event: event.impact_score >= 8.0,
                'priority': AlertPriority.HIGH,
                'title_template': 'High Impact Alert: {supplier_name}',
                'message_template': 'Potential high business impact (score: {impact_score}/10) from {supplier_name}.',
                'threshold': 'Impact Score >= 8.0'
            },
            'multiple_risk_types': {
                'condition': lambda events: len(set(e.event_type for e in events)) >= 3,
                'priority': AlertPriority.MEDIUM,
                'title_template': 'Multiple Risk Types: {supplier_name}',
                'message_template': 'Multiple risk types detected for {supplier_name}: {risk_types}.',
                'threshold': 'Risk Types >= 3'
            },
            'geopolitical_critical': {
                'condition': lambda event: (
                    event.event_type == 'geopolitical' and 
                    event.severity == RiskLevel.CRITICAL
                ),
                'priority': AlertPriority.CRITICAL,
                'title_template': 'Critical Geopolitical Risk: {supplier_name}',
                'message_template': 'Critical geopolitical risk detected for {supplier_name} in {geographic_scope}.',
                'threshold': 'Geopolitical + Critical Severity'
            },
            'financial_distress': {
                'condition': lambda event: (
                    event.event_type == 'financial' and 
                    event.probability >= 0.7
                ),
                'priority': AlertPriority.HIGH,
                'title_template': 'Financial Distress Alert: {supplier_name}',
                'message_template': 'Financial distress indicators detected for {supplier_name}.',
                'threshold': 'Financial Risk + Probability >= 70%'
            },
            'cyber_security_breach': {
                'condition': lambda event: event.event_type == 'cyber',
                'priority': AlertPriority.HIGH,
                'title_template': 'Cybersecurity Alert: {supplier_name}',
                'message_template': 'Cybersecurity risk detected for {supplier_name}.',
                'threshold': 'Cyber Risk Event'
            }
        }
    
    def _initialize_escalation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize escalation rules for unacknowledged alerts."""
        return {
            'critical_unacknowledged': {
                'condition': lambda alert: (
                    alert.priority == AlertPriority.CRITICAL and
                    alert.status == AlertStatus.ACTIVE and
                    (datetime.now() - alert.created_at).total_seconds() > 3600  # 1 hour
                ),
                'escalation_hours': 1,
                'max_escalations': 3
            },
            'high_unacknowledged': {
                'condition': lambda alert: (
                    alert.priority == AlertPriority.HIGH and
                    alert.status == AlertStatus.ACTIVE and
                    (datetime.now() - alert.created_at).total_seconds() > 7200  # 2 hours
                ),
                'escalation_hours': 2,
                'max_escalations': 2
            },
            'medium_unacknowledged': {
                'condition': lambda alert: (
                    alert.priority == AlertPriority.MEDIUM and
                    alert.status == AlertStatus.ACTIVE and
                    (datetime.now() - alert.created_at).total_seconds() > 14400  # 4 hours
                ),
                'escalation_hours': 4,
                'max_escalations': 1
            }
        }
    
    def process_risk_events(
        self, 
        risk_events: List[RiskEvent], 
        suppliers: List[Supplier]
    ) -> List[RiskAlert]:
        """
        Process risk events and generate alerts based on thresholds.
        
        Args:
            risk_events: List of risk events to process
            suppliers: List of suppliers for context
            
        Returns:
            List of generated alerts
        """
        new_alerts = []
        supplier_map = {s.id: s for s in suppliers}
        
        # Group events by supplier for multi-event rules
        events_by_supplier = defaultdict(list)
        for event in risk_events:
            events_by_supplier[event.supplier_id].append(event)
        
        # Process individual event rules
        for event in risk_events:
            supplier = supplier_map.get(event.supplier_id)
            if not supplier:
                continue
            
            # Check each alert rule
            for rule_name, rule in self.alert_rules.items():
                if rule_name == 'multiple_risk_types':
                    continue  # Handle separately
                
                if rule['condition'](event):
                    alert = self._create_alert_from_event(
                        event, supplier, rule_name, rule
                    )
                    if alert and alert.alert_id not in self.alerts:
                        new_alerts.append(alert)
                        self.alerts[alert.alert_id] = alert
        
        # Process multi-event rules
        for supplier_id, supplier_events in events_by_supplier.items():
            supplier = supplier_map.get(supplier_id)
            if not supplier or len(supplier_events) < 2:
                continue
            
            # Check multiple risk types rule
            rule = self.alert_rules['multiple_risk_types']
            if rule['condition'](supplier_events):
                alert = self._create_multi_event_alert(
                    supplier_events, supplier, 'multiple_risk_types', rule
                )
                if alert and alert.alert_id not in self.alerts:
                    new_alerts.append(alert)
                    self.alerts[alert.alert_id] = alert
        
        # Add to dashboard notifications
        for alert in new_alerts:
            self._add_dashboard_notification(alert)
        
        logger.info(f"Generated {len(new_alerts)} new alerts from {len(risk_events)} risk events")
        return new_alerts
    
    def _create_alert_from_event(
        self, 
        event: RiskEvent, 
        supplier: Supplier, 
        rule_name: str, 
        rule: Dict[str, Any]
    ) -> Optional[RiskAlert]:
        """Create an alert from a single risk event."""
        try:
            alert_id = f"ALERT_{event.event_id}_{rule_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Format title and message
            context = {
                'supplier_name': supplier.name,
                'severity': event.severity,
                'probability': event.probability,
                'impact_score': event.impact_score,
                'geographic_scope': event.geographic_scope or supplier.country,
                'event_type': event.event_type
            }
            
            title = rule['title_template'].format(**context)
            message = rule['message_template'].format(**context)
            
            # Generate recommended actions
            recommended_actions = self._generate_alert_actions(event, supplier, rule_name)
            
            alert = RiskAlert(
                alert_id=alert_id,
                event_id=event.event_id,
                supplier_id=supplier.id,
                alert_type=rule_name,
                priority=rule['priority'],
                title=title,
                message=message,
                threshold_breached=rule['threshold'],
                recommended_actions=recommended_actions
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert from event {event.event_id}: {e}")
            return None
    
    def _create_multi_event_alert(
        self, 
        events: List[RiskEvent], 
        supplier: Supplier, 
        rule_name: str, 
        rule: Dict[str, Any]
    ) -> Optional[RiskAlert]:
        """Create an alert from multiple risk events."""
        try:
            # Use the most recent event as primary
            primary_event = max(events, key=lambda e: e.detected_at)
            
            alert_id = f"ALERT_MULTI_{supplier.id}_{rule_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Format context
            risk_types = list(set(e.event_type for e in events))
            context = {
                'supplier_name': supplier.name,
                'risk_types': ', '.join(risk_types)
            }
            
            title = rule['title_template'].format(**context)
            message = rule['message_template'].format(**context)
            
            # Generate comprehensive recommended actions
            recommended_actions = []
            for event in events:
                actions = self._generate_alert_actions(event, supplier, rule_name)
                recommended_actions.extend(actions)
            
            # Remove duplicates while preserving order
            recommended_actions = list(dict.fromkeys(recommended_actions))
            
            alert = RiskAlert(
                alert_id=alert_id,
                event_id=primary_event.event_id,
                supplier_id=supplier.id,
                alert_type=rule_name,
                priority=rule['priority'],
                title=title,
                message=message,
                threshold_breached=rule['threshold'],
                recommended_actions=recommended_actions[:8]  # Limit to 8 actions
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create multi-event alert for supplier {supplier.id}: {e}")
            return None
    
    def _generate_alert_actions(
        self, 
        event: RiskEvent, 
        supplier: Supplier, 
        rule_name: str
    ) -> List[str]:
        """Generate recommended actions for an alert."""
        actions = []
        
        # Base actions from event
        actions.extend(event.mitigation_actions[:3])  # Top 3 from event
        
        # Rule-specific actions
        rule_actions = {
            'high_risk_supplier': [
                'Review supplier contract terms',
                'Activate enhanced monitoring protocols',
                'Assess alternative supplier options'
            ],
            'critical_probability': [
                'Implement immediate contingency plans',
                'Notify key stakeholders',
                'Accelerate backup supplier qualification'
            ],
            'high_impact_score': [
                'Assess business continuity plans',
                'Review inventory levels and buffers',
                'Coordinate with operations team'
            ],
            'geopolitical_critical': [
                'Monitor political developments',
                'Engage government relations team',
                'Review force majeure clauses'
            ],
            'financial_distress': [
                'Review supplier financial health',
                'Consider payment term adjustments',
                'Evaluate supplier support options'
            ],
            'cyber_security_breach': [
                'Assess data security protocols',
                'Review incident response plans',
                'Evaluate cyber insurance coverage'
            ]
        }
        
        actions.extend(rule_actions.get(rule_name, []))
        
        # Priority-based actions
        if event.severity == RiskLevel.CRITICAL:
            actions.extend([
                'Escalate to executive leadership',
                'Activate crisis management team'
            ])
        
        # Remove duplicates and limit
        return list(dict.fromkeys(actions))[:6]
    
    def _add_dashboard_notification(self, alert: RiskAlert):
        """Add alert to dashboard notifications queue."""
        notification = {
            'id': alert.alert_id,
            'type': 'alert',
            'priority': alert.priority,
            'title': alert.title,
            'message': alert.message,
            'timestamp': alert.created_at.isoformat(),
            'supplier_id': alert.supplier_id,
            'event_id': alert.event_id,
            'actions': alert.recommended_actions[:3]  # Top 3 for dashboard
        }
        
        self.dashboard_notifications.appendleft(notification)
    
    def get_active_alerts(self, priority_filter: Optional[str] = None) -> List[RiskAlert]:
        """Get all active alerts, optionally filtered by priority."""
        alerts = [
            alert for alert in self.alerts.values() 
            if alert.status == AlertStatus.ACTIVE
        ]
        
        if priority_filter:
            alerts = [alert for alert in alerts if alert.priority == priority_filter]
        
        # Sort by priority and creation time
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        alerts.sort(key=lambda a: (priority_order.get(a.priority, 4), a.created_at))
        
        return alerts
    
    def get_dashboard_notifications(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent notifications for dashboard display."""
        return list(self.dashboard_notifications)[:limit]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            success = self.alerts[alert_id].acknowledge(acknowledged_by)
            if success:
                # Add acknowledgment notification
                self._add_acknowledgment_notification(self.alerts[alert_id], acknowledged_by)
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return success
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            success = self.alerts[alert_id].resolve()
            if success:
                # Move to history
                self.alert_history.appendleft(self.alerts[alert_id])
                logger.info(f"Alert {alert_id} resolved")
            return success
        return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert."""
        if alert_id in self.alerts:
            success = self.alerts[alert_id].dismiss()
            if success:
                # Move to history
                self.alert_history.appendleft(self.alerts[alert_id])
                logger.info(f"Alert {alert_id} dismissed")
            return success
        return False
    
    def _add_acknowledgment_notification(self, alert: RiskAlert, acknowledged_by: str):
        """Add acknowledgment notification to dashboard."""
        notification = {
            'id': f"ACK_{alert.alert_id}",
            'type': 'acknowledgment',
            'priority': 'info',
            'title': 'Alert Acknowledged',
            'message': f'Alert "{alert.title}" acknowledged by {acknowledged_by}',
            'timestamp': datetime.now().isoformat(),
            'supplier_id': alert.supplier_id,
            'event_id': alert.event_id
        }
        
        self.dashboard_notifications.appendleft(notification)
    
    def check_escalations(self) -> List[RiskAlert]:
        """Check for alerts that need escalation."""
        escalated_alerts = []
        
        for alert in self.alerts.values():
            for rule_name, rule in self.escalation_rules.items():
                if (rule['condition'](alert) and 
                    alert.escalation_level < rule['max_escalations']):
                    
                    if alert.escalate():
                        escalated_alerts.append(alert)
                        self._add_escalation_notification(alert)
                        logger.warning(f"Alert {alert.alert_id} escalated to level {alert.escalation_level}")
        
        return escalated_alerts
    
    def _add_escalation_notification(self, alert: RiskAlert):
        """Add escalation notification to dashboard."""
        notification = {
            'id': f"ESC_{alert.alert_id}_{alert.escalation_level}",
            'type': 'escalation',
            'priority': 'warning',
            'title': 'Alert Escalated',
            'message': f'Alert "{alert.title}" escalated to level {alert.escalation_level}',
            'timestamp': datetime.now().isoformat(),
            'supplier_id': alert.supplier_id,
            'event_id': alert.event_id,
            'escalation_level': alert.escalation_level
        }
        
        self.dashboard_notifications.appendleft(notification)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about alerts."""
        active_alerts = self.get_active_alerts()
        
        # Count by priority
        priority_counts = defaultdict(int)
        for alert in active_alerts:
            priority_counts[alert.priority] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in active_alerts:
            type_counts[alert.alert_type] += 1
        
        # Count by status
        status_counts = defaultdict(int)
        for alert in self.alerts.values():
            status_counts[alert.status] += 1
        
        # Calculate response times
        acknowledged_alerts = [
            alert for alert in self.alerts.values() 
            if alert.acknowledged_at
        ]
        
        avg_response_time = 0
        if acknowledged_alerts:
            response_times = [
                (alert.acknowledged_at - alert.created_at).total_seconds() / 3600
                for alert in acknowledged_alerts
            ]
            avg_response_time = sum(response_times) / len(response_times)
        
        return {
            'total_active_alerts': len(active_alerts),
            'priority_distribution': dict(priority_counts),
            'type_distribution': dict(type_counts),
            'status_distribution': dict(status_counts),
            'average_response_time_hours': round(avg_response_time, 2),
            'escalated_alerts': len([a for a in active_alerts if a.escalation_level > 0]),
            'overdue_alerts': len([a for a in active_alerts if a.is_overdue()]),
            'total_notifications': len(self.dashboard_notifications)
        }
    
    def get_supplier_alerts(self, supplier_id: str) -> List[RiskAlert]:
        """Get all alerts for a specific supplier."""
        return [
            alert for alert in self.alerts.values()
            if alert.supplier_id == supplier_id
        ]
    
    def clear_old_notifications(self, hours: int = 24):
        """Clear old dashboard notifications."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter notifications
        filtered_notifications = deque(maxlen=50)
        for notification in self.dashboard_notifications:
            notification_time = datetime.fromisoformat(notification['timestamp'])
            if notification_time > cutoff_time:
                filtered_notifications.append(notification)
        
        self.dashboard_notifications = filtered_notifications
        logger.info(f"Cleared notifications older than {hours} hours")
    
    def export_alerts_summary(self) -> Dict[str, Any]:
        """Export summary of all alerts for reporting."""
        return {
            'export_timestamp': datetime.now().isoformat(),
            'active_alerts': [alert.to_dict() for alert in self.get_active_alerts()],
            'recent_notifications': list(self.dashboard_notifications),
            'statistics': self.get_alert_statistics(),
            'alert_rules': {
                name: {
                    'priority': rule['priority'],
                    'threshold': rule['threshold']
                }
                for name, rule in self.alert_rules.items()
            }
        }
    
    def add_custom_alert_rule(
        self, 
        rule_name: str, 
        condition: Callable, 
        priority: AlertPriority,
        title_template: str, 
        message_template: str, 
        threshold_description: str
    ) -> bool:
        """Add a custom alert rule."""
        try:
            self.alert_rules[rule_name] = {
                'condition': condition,
                'priority': priority,
                'title_template': title_template,
                'message_template': message_template,
                'threshold': threshold_description
            }
            logger.info(f"Added custom alert rule: {rule_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add custom alert rule {rule_name}: {e}")
            return False
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False