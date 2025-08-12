"""
Streamlit notification components for risk alerts and system notifications.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

from models.risk_alert_system import RiskAlertSystem, RiskAlert, AlertPriority, AlertStatus


class StreamlitNotificationManager:
    """
    Manages notifications and alerts display in Streamlit dashboard.
    """
    
    def __init__(self, alert_system: RiskAlertSystem):
        """
        Initialize notification manager.
        
        Args:
            alert_system: RiskAlertSystem instance
        """
        self.alert_system = alert_system
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state for notifications."""
        if 'notification_settings' not in st.session_state:
            st.session_state.notification_settings = {
                'show_notifications': True,
                'auto_dismiss_time': 10,  # seconds
                'priority_filter': 'all',
                'max_notifications': 10
            }
        
        if 'dismissed_notifications' not in st.session_state:
            st.session_state.dismissed_notifications = set()
        
        if 'acknowledged_alerts' not in st.session_state:
            st.session_state.acknowledged_alerts = set()
    
    def render_notification_banner(self) -> bool:
        """
        Render notification banner at top of dashboard.
        
        Returns:
            True if notifications are displayed, False otherwise
        """
        if not st.session_state.notification_settings['show_notifications']:
            return False
        
        # Get active alerts
        active_alerts = self.alert_system.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.priority == AlertPriority.CRITICAL]
        high_alerts = [a for a in active_alerts if a.priority == AlertPriority.HIGH]
        
        if not active_alerts:
            return False
        
        # Create notification container
        notification_container = st.container()
        
        with notification_container:
            # Critical alerts banner
            if critical_alerts:
                self._render_critical_alert_banner(critical_alerts)
            
            # High priority alerts
            if high_alerts and len(critical_alerts) == 0:
                self._render_high_alert_banner(high_alerts)
            
            # General notification area
            self._render_notification_area()
        
        return True
    
    def _render_critical_alert_banner(self, critical_alerts: List[RiskAlert]):
        """Render critical alert banner."""
        alert_count = len(critical_alerts)
        
        st.error(f"""
        🚨 **CRITICAL ALERTS ({alert_count})** - Immediate attention required!
        
        {critical_alerts[0].title}
        
        **Actions Required:** {', '.join(critical_alerts[0].recommended_actions[:2])}
        """)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🔍 View Details", key=f"critical_details_{critical_alerts[0].alert_id}"):
                self._show_alert_details(critical_alerts[0])
        
        with col2:
            if st.button("✅ Acknowledge", key=f"critical_ack_{critical_alerts[0].alert_id}"):
                self._acknowledge_alert(critical_alerts[0].alert_id)
        
        with col3:
            if st.button("❌ Dismiss", key=f"critical_dismiss_{critical_alerts[0].alert_id}"):
                self._dismiss_alert(critical_alerts[0].alert_id)
    
    def _render_high_alert_banner(self, high_alerts: List[RiskAlert]):
        """Render high priority alert banner."""
        alert_count = len(high_alerts)
        
        st.warning(f"""
        ⚠️ **HIGH PRIORITY ALERTS ({alert_count})** - Action recommended
        
        {high_alerts[0].title}
        """)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🔍 View Details", key=f"high_details_{high_alerts[0].alert_id}"):
                self._show_alert_details(high_alerts[0])
        
        with col2:
            if st.button("✅ Acknowledge", key=f"high_ack_{high_alerts[0].alert_id}"):
                self._acknowledge_alert(high_alerts[0].alert_id)
        
        with col3:
            if st.button("❌ Dismiss", key=f"high_dismiss_{high_alerts[0].alert_id}"):
                self._dismiss_alert(high_alerts[0].alert_id)
    
    def _render_notification_area(self):
        """Render general notification area."""
        notifications = self.alert_system.get_dashboard_notifications(
            limit=st.session_state.notification_settings['max_notifications']
        )
        
        if not notifications:
            return
        
        # Filter out dismissed notifications
        active_notifications = [
            n for n in notifications 
            if n['id'] not in st.session_state.dismissed_notifications
        ]
        
        if not active_notifications:
            return
        
        # Notification area header
        with st.expander(f"📢 Recent Notifications ({len(active_notifications)})", expanded=False):
            for notification in active_notifications:
                self._render_notification_item(notification)
    
    def _render_notification_item(self, notification: Dict[str, Any]):
        """Render individual notification item."""
        notification_id = notification['id']
        notification_type = notification['type']
        priority = notification['priority']
        
        # Priority styling
        priority_colors = {
            'critical': '🔴',
            'high': '🟠', 
            'medium': '🟡',
            'low': '🟢',
            'info': '🔵',
            'warning': '🟠'
        }
        
        priority_icon = priority_colors.get(priority, '🔵')
        
        # Notification content
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"""
            {priority_icon} **{notification['title']}**  
            {notification['message']}  
            *{datetime.fromisoformat(notification['timestamp']).strftime('%H:%M:%S')}*
            """)
            
            # Show actions if available
            if 'actions' in notification and notification['actions']:
                with st.expander("Recommended Actions", expanded=False):
                    for action in notification['actions']:
                        st.markdown(f"• {action}")
        
        with col2:
            if st.button("❌", key=f"dismiss_{notification_id}", help="Dismiss notification"):
                st.session_state.dismissed_notifications.add(notification_id)
                st.rerun()
    
    def render_alert_management_panel(self):
        """Render alert management panel in sidebar or dedicated section."""
        st.subheader("🚨 Alert Management")
        
        # Alert statistics
        stats = self.alert_system.get_alert_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Alerts", stats['total_active_alerts'])
        
        with col2:
            st.metric("Escalated", stats['escalated_alerts'])
        
        with col3:
            st.metric("Overdue", stats['overdue_alerts'])
        
        # Alert filters
        st.subheader("🔍 Filter Alerts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            priority_filter = st.selectbox(
                "Priority",
                options=['all', 'critical', 'high', 'medium', 'low'],
                key='alert_priority_filter'
            )
        
        with col2:
            status_filter = st.selectbox(
                "Status", 
                options=['active', 'acknowledged', 'resolved', 'dismissed'],
                key='alert_status_filter'
            )
        
        # Get filtered alerts
        if priority_filter == 'all':
            alerts = self.alert_system.get_active_alerts()
        else:
            alerts = self.alert_system.get_active_alerts(priority_filter)
        
        # Display alerts table
        if alerts:
            self._render_alerts_table(alerts)
        else:
            st.info("No alerts match the current filters.")
        
        # Bulk actions
        st.subheader("⚡ Bulk Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Acknowledge All High"):
                self._bulk_acknowledge_alerts('high')
        
        with col2:
            if st.button("🔄 Check Escalations"):
                escalated = self.alert_system.check_escalations()
                if escalated:
                    st.success(f"Escalated {len(escalated)} alerts")
                else:
                    st.info("No alerts need escalation")
        
        with col3:
            if st.button("🧹 Clear Old Notifications"):
                self.alert_system.clear_old_notifications(hours=24)
                st.success("Cleared old notifications")
    
    def _render_alerts_table(self, alerts: List[RiskAlert]):
        """Render alerts in a table format."""
        # Prepare data for table
        table_data = []
        
        for alert in alerts:
            table_data.append({
                'ID': alert.alert_id[-8:],  # Last 8 characters
                'Priority': alert.priority.value.upper(),
                'Title': alert.title,
                'Supplier': alert.supplier_id,
                'Created': alert.created_at.strftime('%m/%d %H:%M'),
                'Status': alert.status.value.upper(),
                'Escalation': alert.escalation_level,
                'Actions': len(alert.recommended_actions)
            })
        
        df = pd.DataFrame(table_data)
        
        # Configure column display
        column_config = {
            'Priority': st.column_config.SelectboxColumn(
                'Priority',
                options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            ),
            'Status': st.column_config.SelectboxColumn(
                'Status',
                options=['ACTIVE', 'ACKNOWLEDGED', 'RESOLVED', 'DISMISSED']
            ),
            'Escalation': st.column_config.NumberColumn(
                'Escalation Level',
                min_value=0,
                max_value=3
            )
        }
        
        # Display table with selection
        selected_rows = st.dataframe(
            df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row"
        )
        
        # Actions for selected alerts
        if hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
            selected_alert_ids = [alerts[i].alert_id for i in selected_rows.selection.rows]
            
            st.subheader("Actions for Selected Alerts")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Acknowledge Selected"):
                    for alert_id in selected_alert_ids:
                        self._acknowledge_alert(alert_id)
            
            with col2:
                if st.button("✅ Resolve Selected"):
                    for alert_id in selected_alert_ids:
                        self._resolve_alert(alert_id)
            
            with col3:
                if st.button("❌ Dismiss Selected"):
                    for alert_id in selected_alert_ids:
                        self._dismiss_alert(alert_id)
    
    def _show_alert_details(self, alert: RiskAlert):
        """Show detailed alert information in modal/expander."""
        with st.expander(f"Alert Details: {alert.title}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Alert ID:** {alert.alert_id}  
                **Event ID:** {alert.event_id}  
                **Supplier ID:** {alert.supplier_id}  
                **Priority:** {alert.priority.value.upper()}  
                **Status:** {alert.status.value.upper()}  
                **Created:** {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
                """)
            
            with col2:
                st.markdown(f"""
                **Alert Type:** {alert.alert_type}  
                **Threshold Breached:** {alert.threshold_breached}  
                **Escalation Level:** {alert.escalation_level}  
                **Notifications Sent:** {alert.notification_count}  
                """)
            
            st.markdown(f"**Message:** {alert.message}")
            
            if alert.recommended_actions:
                st.markdown("**Recommended Actions:**")
                for i, action in enumerate(alert.recommended_actions, 1):
                    st.markdown(f"{i}. {action}")
    
    def _acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        user = st.session_state.get('user_name', 'dashboard_user')
        success = self.alert_system.acknowledge_alert(alert_id, user)
        
        if success:
            st.session_state.acknowledged_alerts.add(alert_id)
            st.success(f"Alert {alert_id[-8:]} acknowledged")
            st.rerun()
        else:
            st.error("Failed to acknowledge alert")
    
    def _resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        success = self.alert_system.resolve_alert(alert_id)
        
        if success:
            st.success(f"Alert {alert_id[-8:]} resolved")
            st.rerun()
        else:
            st.error("Failed to resolve alert")
    
    def _dismiss_alert(self, alert_id: str):
        """Dismiss an alert."""
        success = self.alert_system.dismiss_alert(alert_id)
        
        if success:
            st.success(f"Alert {alert_id[-8:]} dismissed")
            st.rerun()
        else:
            st.error("Failed to dismiss alert")
    
    def _bulk_acknowledge_alerts(self, priority: str):
        """Acknowledge all alerts of a given priority."""
        alerts = self.alert_system.get_active_alerts(priority)
        user = st.session_state.get('user_name', 'dashboard_user')
        
        acknowledged_count = 0
        for alert in alerts:
            if self.alert_system.acknowledge_alert(alert.alert_id, user):
                acknowledged_count += 1
        
        if acknowledged_count > 0:
            st.success(f"Acknowledged {acknowledged_count} {priority} priority alerts")
            st.rerun()
        else:
            st.info(f"No {priority} priority alerts to acknowledge")
    
    def render_notification_settings(self):
        """Render notification settings panel."""
        st.subheader("⚙️ Notification Settings")
        
        # Current settings
        settings = st.session_state.notification_settings
        
        # Show notifications toggle
        show_notifications = st.checkbox(
            "Show Notifications",
            value=settings['show_notifications'],
            help="Enable/disable notification display"
        )
        
        # Auto dismiss time
        auto_dismiss_time = st.slider(
            "Auto Dismiss Time (seconds)",
            min_value=5,
            max_value=60,
            value=settings['auto_dismiss_time'],
            help="Time before notifications auto-dismiss"
        )
        
        # Priority filter
        priority_filter = st.selectbox(
            "Default Priority Filter",
            options=['all', 'critical', 'high', 'medium', 'low'],
            index=['all', 'critical', 'high', 'medium', 'low'].index(settings['priority_filter']),
            help="Default priority filter for notifications"
        )
        
        # Max notifications
        max_notifications = st.number_input(
            "Max Notifications to Display",
            min_value=5,
            max_value=50,
            value=settings['max_notifications'],
            help="Maximum number of notifications to show"
        )
        
        # Update settings
        if st.button("💾 Save Settings"):
            st.session_state.notification_settings = {
                'show_notifications': show_notifications,
                'auto_dismiss_time': auto_dismiss_time,
                'priority_filter': priority_filter,
                'max_notifications': max_notifications
            }
            st.success("Settings saved successfully!")
            st.rerun()
        
        # Reset settings
        if st.button("🔄 Reset to Defaults"):
            st.session_state.notification_settings = {
                'show_notifications': True,
                'auto_dismiss_time': 10,
                'priority_filter': 'all',
                'max_notifications': 10
            }
            st.success("Settings reset to defaults!")
            st.rerun()
    
    def render_alert_history(self, days: int = 7):
        """Render alert history for the specified number of days."""
        st.subheader(f"📜 Alert History ({days} days)")
        
        # Get alert statistics
        stats = self.alert_system.get_alert_statistics()
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", stats.get('total_active_alerts', 0))
        
        with col2:
            avg_response = stats.get('average_response_time_hours', 0)
            st.metric("Avg Response Time", f"{avg_response:.1f}h")
        
        with col3:
            resolution_rate = stats.get('resolution_rate', 0)
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        
        with col4:
            escalated = stats.get('escalated_alerts', 0)
            st.metric("Escalated Alerts", escalated)
        
        # Priority distribution chart
        if 'priority_distribution' in stats:
            priority_data = stats['priority_distribution']
            
            if priority_data:
                df_priority = pd.DataFrame(
                    list(priority_data.items()),
                    columns=['Priority', 'Count']
                )
                
                st.subheader("📊 Alert Distribution by Priority")
                st.bar_chart(df_priority.set_index('Priority'))
        
        # Type distribution
        if 'type_distribution' in stats:
            type_data = stats['type_distribution']
            
            if type_data:
                df_type = pd.DataFrame(
                    list(type_data.items()),
                    columns=['Alert Type', 'Count']
                )
                
                st.subheader("📈 Alert Distribution by Type")
                st.bar_chart(df_type.set_index('Alert Type'))
    
    def export_alert_data(self) -> str:
        """Export alert data as JSON string."""
        alert_summary = self.alert_system.export_alerts_summary()
        return json.dumps(alert_summary, indent=2, default=str)
    
    def render_export_section(self):
        """Render alert data export section."""
        st.subheader("📤 Export Alert Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Copy Alert Summary"):
                alert_data = self.export_alert_data()
                st.text_area("Alert Data (JSON)", value=alert_data, height=200)
        
        with col2:
            if st.button("💾 Download Alert Data"):
                alert_data = self.export_alert_data()
                st.download_button(
                    "⬇️ Download JSON",
                    data=alert_data,
                    file_name=f"alert_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )