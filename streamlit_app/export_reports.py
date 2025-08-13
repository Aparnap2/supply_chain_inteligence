"""
Export and reporting functionality for the Supply Chain Intelligence Dashboard.
"""

import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import base64
from pathlib import Path

# Add project root to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.analysis_result import AnalysisResult
from models.supplier import Supplier
from models.risk_event import RiskEvent
from models.ml_prediction import MLPrediction


class ExportReportManager:
    """Manager for export and reporting functionality."""
    
    def __init__(self):
        """Initialize export manager."""
        self.export_formats = ["JSON", "CSV", "PDF"]
        self.report_types = [
            "Executive Summary",
            "Detailed Risk Analysis", 
            "Supplier Assessment",
            "ML Model Report",
            "Complete Analysis"
        ]
    
    def render_export_section(self, analysis_result: Optional[AnalysisResult] = None):
        """Render export and reporting section."""
        
        st.header("📤 Export & Reporting")
        
        if analysis_result is None:
            st.warning("No analysis results available for export. Please run an analysis first.")
            return
        
        # Export options
        export_tabs = st.tabs([
            "📊 Quick Export",
            "📋 Custom Reports", 
            "⏰ Scheduled Reports",
            "📧 Report Delivery"
        ])
        
        with export_tabs[0]:
            self._render_quick_export(analysis_result)
        
        with export_tabs[1]:
            self._render_custom_reports(analysis_result)
        
        with export_tabs[2]:
            self._render_scheduled_reports(analysis_result)
        
        with export_tabs[3]:
            self._render_report_delivery(analysis_result)
    
    def _render_quick_export(self, analysis_result: AnalysisResult):
        """Render quick export options."""
        
        st.subheader("📊 Quick Export")
        st.write("Export current analysis results in various formats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export JSON", use_container_width=True):
                json_data = self.export_to_json(analysis_result)
                self._download_file(
                    json_data,
                    f"supply_chain_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        with col2:
            if st.button("📊 Export CSV", use_container_width=True):
                csv_data = self.export_to_csv(analysis_result)
                self._download_file(
                    csv_data,
                    f"supply_chain_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        with col3:
            if st.button("📑 Export PDF", use_container_width=True):
                pdf_data = self.export_to_pdf(analysis_result)
                if pdf_data:
                    self._download_file(
                        pdf_data,
                        f"supply_chain_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf"
                    )
                else:
                    st.error("PDF export not available. Please install required dependencies.")
        
        # Export summary
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Suppliers", len(analysis_result.suppliers))
        with col2:
            st.metric("Risk Events", len(analysis_result.risk_events))
        with col3:
            st.metric("ML Predictions", len(analysis_result.ml_predictions))
        with col4:
            st.metric("Data Quality", f"{analysis_result.data_quality_score:.1%}")
    
    def _render_custom_reports(self, analysis_result: AnalysisResult):
        """Render custom report generation."""
        
        st.subheader("📋 Custom Report Generator")
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                options=self.report_types,
                help="Select the type of report to generate"
            )
            
            export_format = st.selectbox(
                "Export Format",
                options=self.export_formats,
                help="Choose export format"
            )
        
        with col2:
            include_charts = st.checkbox("Include Charts", value=True)
            include_raw_data = st.checkbox("Include Raw Data", value=False)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
            
            date_range = st.date_input(
                "Report Date Range",
                value=[datetime.now().date()],
                help="Select date range for the report"
            )
        
        # Report sections
        st.subheader("Report Sections")
        
        sections = {
            "Executive Summary": st.checkbox("Executive Summary", value=True),
            "Risk Analysis": st.checkbox("Risk Analysis", value=True),
            "Supplier Overview": st.checkbox("Supplier Overview", value=True),
            "ML Insights": st.checkbox("ML Model Insights", value=True),
            "Geographic Analysis": st.checkbox("Geographic Analysis", value=False),
            "Industry Analysis": st.checkbox("Industry Analysis", value=False),
            "Recommendations": st.checkbox("Strategic Recommendations", value=True),
            "Appendices": st.checkbox("Data Appendices", value=False)
        }
        
        # Generate custom report
        if st.button("🔄 Generate Custom Report", type="primary"):
            with st.spinner("Generating custom report..."):
                report_config = {
                    "report_type": report_type,
                    "export_format": export_format,
                    "include_charts": include_charts,
                    "include_raw_data": include_raw_data,
                    "include_recommendations": include_recommendations,
                    "sections": {k: v for k, v in sections.items() if v},
                    "date_range": date_range
                }
                
                report_data = self.generate_custom_report(analysis_result, report_config)
                
                if report_data:
                    filename = f"custom_report_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    if export_format == "JSON":
                        self._download_file(report_data, f"{filename}.json", "application/json")
                    elif export_format == "CSV":
                        self._download_file(report_data, f"{filename}.csv", "text/csv")
                    elif export_format == "PDF":
                        self._download_file(report_data, f"{filename}.pdf", "application/pdf")
                    
                    st.success("Custom report generated successfully!")
                else:
                    st.error("Failed to generate custom report.")
    
    def _render_scheduled_reports(self, analysis_result: AnalysisResult):
        """Render scheduled reporting options."""
        
        st.subheader("⏰ Scheduled Reports")
        st.info("📝 Note: Scheduled reporting requires backend integration (not implemented in demo)")
        
        # Schedule configuration
        col1, col2 = st.columns(2)
        
        with col1:
            schedule_type = st.selectbox(
                "Schedule Type",
                options=["Daily", "Weekly", "Monthly", "Quarterly"],
                help="How often to generate reports"
            )
            
            report_format = st.selectbox(
                "Report Format",
                options=["PDF", "JSON", "CSV"],
                help="Format for scheduled reports"
            )
        
        with col2:
            schedule_time = st.time_input(
                "Schedule Time",
                value=datetime.now().time(),
                help="Time to generate reports"
            )
            
            recipients = st.text_area(
                "Email Recipients",
                placeholder="Enter email addresses separated by commas",
                help="Who should receive the scheduled reports"
            )
        
        # Active schedules (demo data)
        st.subheader("Active Schedules")
        
        schedule_data = [
            {
                "Schedule ID": "SCH_001",
                "Type": "Weekly",
                "Format": "PDF",
                "Recipients": "manager@company.com",
                "Next Run": "2024-01-15 09:00",
                "Status": "Active"
            },
            {
                "Schedule ID": "SCH_002", 
                "Type": "Monthly",
                "Format": "JSON",
                "Recipients": "analytics@company.com",
                "Next Run": "2024-02-01 08:00",
                "Status": "Active"
            }
        ]
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)
        
        # Schedule management buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add Schedule"):
                st.success("Schedule configuration saved (demo)")
        
        with col2:
            if st.button("⏸️ Pause All"):
                st.info("All schedules paused (demo)")
        
        with col3:
            if st.button("🗑️ Delete Selected"):
                st.warning("Selected schedules deleted (demo)")
    
    def _render_report_delivery(self, analysis_result: AnalysisResult):
        """Render report delivery options."""
        
        st.subheader("📧 Report Delivery")
        
        # Delivery methods
        delivery_method = st.radio(
            "Delivery Method",
            options=["Email", "Cloud Storage", "API Webhook", "FTP/SFTP"],
            help="How to deliver generated reports"
        )
        
        if delivery_method == "Email":
            st.text_input("SMTP Server", placeholder="smtp.company.com")
            st.text_input("Email From", placeholder="reports@company.com")
            st.text_area("Default Recipients", placeholder="Enter email addresses")
            
        elif delivery_method == "Cloud Storage":
            cloud_provider = st.selectbox("Provider", ["AWS S3", "Google Cloud", "Azure Blob"])
            st.text_input("Bucket/Container", placeholder="supply-chain-reports")
            st.text_input("Access Key", type="password")
            
        elif delivery_method == "API Webhook":
            st.text_input("Webhook URL", placeholder="https://api.company.com/reports")
            st.text_input("API Key", type="password")
            st.selectbox("HTTP Method", ["POST", "PUT"])
            
        elif delivery_method == "FTP/SFTP":
            st.text_input("Server", placeholder="ftp.company.com")
            st.text_input("Username")
            st.text_input("Password", type="password")
            st.text_input("Remote Path", placeholder="/reports/supply-chain/")
        
        # Test delivery
        if st.button("🧪 Test Delivery"):
            st.info(f"Testing {delivery_method} delivery configuration... (demo)")
            st.success("Delivery test successful!")
        
        # Delivery history
        st.subheader("Delivery History")
        
        delivery_history = [
            {
                "Timestamp": "2024-01-14 09:00:15",
                "Report": "Weekly Risk Summary",
                "Method": "Email",
                "Recipients": "3",
                "Status": "✅ Delivered",
                "Size": "2.3 MB"
            },
            {
                "Timestamp": "2024-01-13 18:30:22", 
                "Report": "Supplier Assessment",
                "Method": "Cloud Storage",
                "Recipients": "S3 Bucket",
                "Status": "✅ Uploaded",
                "Size": "5.7 MB"
            },
            {
                "Timestamp": "2024-01-12 14:15:08",
                "Report": "ML Model Report",
                "Method": "API Webhook",
                "Recipients": "Analytics API",
                "Status": "❌ Failed",
                "Size": "1.2 MB"
            }
        ]
        
        delivery_df = pd.DataFrame(delivery_history)
        st.dataframe(delivery_df, use_container_width=True, hide_index=True)
    
    def export_to_json(self, analysis_result: AnalysisResult) -> bytes:
        """Export analysis result to JSON format."""
        
        try:
            # Convert to dictionary with proper serialization
            export_data = {
                "analysis_metadata": {
                    "analysis_id": analysis_result.analysis_id,
                    "generated_at": analysis_result.generated_at.isoformat(),
                    "overall_risk_score": analysis_result.overall_risk_score,
                    "data_quality_score": analysis_result.data_quality_score,
                    "workflow_status": analysis_result.workflow_status,
                    "processing_time": analysis_result.processing_time
                },
                "suppliers": [supplier.dict() for supplier in analysis_result.suppliers],
                "risk_events": [risk_event.dict() for risk_event in analysis_result.risk_events],
                "ml_predictions": [prediction.dict() for prediction in analysis_result.ml_predictions],
                "recommendations": analysis_result.recommendations,
                "summary_metrics": analysis_result.get_summary_metrics()
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            return json_str.encode('utf-8')
            
        except Exception as e:
            st.error(f"Error exporting to JSON: {str(e)}")
            return b""
    
    def export_to_csv(self, analysis_result: AnalysisResult) -> bytes:
        """Export analysis result to CSV format."""
        
        try:
            # Create multiple CSV sheets in a zip-like format
            csv_data = io.StringIO()
            
            # Suppliers data
            suppliers_data = []
            for supplier in analysis_result.suppliers:
                suppliers_data.append({
                    'Supplier ID': supplier.id,
                    'Name': supplier.name,
                    'Country': supplier.country,
                    'Region': supplier.region,
                    'Industry': supplier.industry,
                    'Tier': supplier.tier,
                    'Criticality Score': supplier.criticality_score,
                    'Financial Health': supplier.financial_health_score,
                    'Annual Revenue': supplier.annual_revenue,
                    'Employee Count': supplier.employee_count,
                    'Website': supplier.website
                })
            
            suppliers_df = pd.DataFrame(suppliers_data)
            
            # Risk events data
            risk_events_data = []
            for risk_event in analysis_result.risk_events:
                risk_events_data.append({
                    'Event ID': risk_event.event_id,
                    'Supplier ID': risk_event.supplier_id,
                    'Event Type': risk_event.event_type,
                    'Severity': risk_event.severity,
                    'Impact Score': risk_event.impact_score,
                    'Probability': risk_event.probability,
                    'Confidence': risk_event.confidence_level,
                    'Description': risk_event.description,
                    'Geographic Scope': risk_event.geographic_scope,
                    'Timeline': risk_event.predicted_timeline,
                    'Status': risk_event.status,
                    'Detected At': risk_event.detected_at.isoformat()
                })
            
            risk_events_df = pd.DataFrame(risk_events_data)
            
            # ML predictions data
            ml_predictions_data = []
            for prediction in analysis_result.ml_predictions:
                ml_predictions_data.append({
                    'Prediction ID': prediction.prediction_id,
                    'Supplier ID': prediction.supplier_id,
                    'Model Version': prediction.model_version,
                    'Prediction Type': prediction.prediction_type,
                    'Prediction': prediction.prediction,
                    'Confidence': prediction.confidence,
                    'Predicted At': prediction.predicted_at.isoformat()
                })
            
            ml_predictions_df = pd.DataFrame(ml_predictions_data)
            
            # Combine all data into one CSV with section headers
            csv_data.write("# Supply Chain Intelligence Analysis Export\n")
            csv_data.write(f"# Generated: {datetime.now().isoformat()}\n")
            csv_data.write(f"# Analysis ID: {analysis_result.analysis_id}\n\n")
            
            csv_data.write("## SUPPLIERS\n")
            suppliers_df.to_csv(csv_data, index=False)
            csv_data.write("\n## RISK EVENTS\n")
            risk_events_df.to_csv(csv_data, index=False)
            csv_data.write("\n## ML PREDICTIONS\n")
            ml_predictions_df.to_csv(csv_data, index=False)
            
            return csv_data.getvalue().encode('utf-8')
            
        except Exception as e:
            st.error(f"Error exporting to CSV: {str(e)}")
            return b""
    
    def export_to_pdf(self, analysis_result: AnalysisResult) -> Optional[bytes]:
        """Export analysis result to PDF format."""
        
        try:
            # Note: This is a simplified PDF export
            # In a real implementation, you would use libraries like reportlab or weasyprint
            
            st.warning("PDF export requires additional dependencies (reportlab, weasyprint). Generating text-based report instead.")
            
            # Generate text-based report
            report_content = self._generate_text_report(analysis_result)
            return report_content.encode('utf-8')
            
        except Exception as e:
            st.error(f"Error exporting to PDF: {str(e)}")
            return None
    
    def generate_custom_report(self, analysis_result: AnalysisResult, 
                             config: Dict[str, Any]) -> Optional[bytes]:
        """Generate custom report based on configuration."""
        
        try:
            report_type = config.get("report_type", "Complete Analysis")
            export_format = config.get("export_format", "JSON")
            sections = config.get("sections", {})
            
            # Build custom report data
            custom_data = {
                "report_metadata": {
                    "report_type": report_type,
                    "generated_at": datetime.now().isoformat(),
                    "analysis_id": analysis_result.analysis_id,
                    "sections_included": list(sections.keys())
                }
            }
            
            # Add requested sections
            if sections.get("Executive Summary"):
                custom_data["executive_summary"] = analysis_result.get_summary_metrics()
            
            if sections.get("Risk Analysis"):
                custom_data["risk_analysis"] = {
                    "risk_events": [re.dict() for re in analysis_result.risk_events],
                    "risk_distribution": analysis_result.get_risk_distribution()
                }
            
            if sections.get("Supplier Overview"):
                custom_data["suppliers"] = [s.dict() for s in analysis_result.suppliers]
            
            if sections.get("ML Insights"):
                custom_data["ml_predictions"] = [mp.dict() for mp in analysis_result.ml_predictions]
            
            if sections.get("Recommendations"):
                custom_data["recommendations"] = analysis_result.recommendations
            
            # Export in requested format
            if export_format == "JSON":
                return json.dumps(custom_data, indent=2, default=str).encode('utf-8')
            elif export_format == "CSV":
                # Convert to CSV format
                return self._convert_to_csv(custom_data)
            elif export_format == "PDF":
                # Generate PDF (simplified)
                return self._generate_text_report(analysis_result).encode('utf-8')
            
        except Exception as e:
            st.error(f"Error generating custom report: {str(e)}")
            return None
    
    def _generate_text_report(self, analysis_result: AnalysisResult) -> str:
        """Generate text-based report."""
        
        report = f"""
SUPPLY CHAIN INTELLIGENCE ANALYSIS REPORT
=========================================

Analysis ID: {analysis_result.analysis_id}
Generated: {analysis_result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Overall Risk Score: {analysis_result.overall_risk_score:.1f}/100
Data Quality Score: {analysis_result.data_quality_score:.1%}

EXECUTIVE SUMMARY
-----------------
Total Suppliers Analyzed: {len(analysis_result.suppliers)}
Risk Events Identified: {len(analysis_result.risk_events)}
High-Risk Suppliers: {len(analysis_result.high_risk_suppliers)}
Critical Risk Events: {len(analysis_result.critical_risk_events)}

SUPPLIERS
---------
"""
        
        for supplier in analysis_result.suppliers:
            report += f"""
{supplier.name} ({supplier.id})
  Country: {supplier.country}
  Industry: {supplier.industry}
  Criticality: {supplier.criticality_score:.1f}/100
  Tier: {supplier.tier.replace('_', ' ').title()}
"""
        
        report += "\nRISK EVENTS\n-----------\n"
        
        for risk_event in analysis_result.risk_events:
            report += f"""
{risk_event.event_id} - {risk_event.event_type.title()} Risk
  Supplier: {risk_event.supplier_id}
  Severity: {risk_event.severity.title()}
  Impact Score: {risk_event.impact_score:.1f}/10
  Probability: {risk_event.probability:.1%}
  Description: {risk_event.description}
"""
        
        report += "\nRECOMMENDATIONS\n---------------\n"
        
        for i, recommendation in enumerate(analysis_result.recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        return report
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> bytes:
        """Convert dictionary data to CSV format."""
        
        csv_output = io.StringIO()
        
        # Write metadata
        csv_output.write("# Custom Report Export\n")
        csv_output.write(f"# Generated: {data['report_metadata']['generated_at']}\n\n")
        
        # Convert each section to CSV
        for section, content in data.items():
            if section == "report_metadata":
                continue
                
            csv_output.write(f"## {section.upper()}\n")
            
            if isinstance(content, list) and content:
                # Convert list of dictionaries to DataFrame
                df = pd.DataFrame(content)
                df.to_csv(csv_output, index=False)
            elif isinstance(content, dict):
                # Convert dictionary to DataFrame
                df = pd.DataFrame([content])
                df.to_csv(csv_output, index=False)
            
            csv_output.write("\n")
        
        return csv_output.getvalue().encode('utf-8')
    
    def _download_file(self, data: bytes, filename: str, mime_type: str):
        """Create download link for file."""
        
        b64_data = base64.b64encode(data).decode()
        
        st.download_button(
            label=f"⬇️ Download {filename}",
            data=data,
            file_name=filename,
            mime=mime_type,
            key=f"download_{filename}_{datetime.now().timestamp()}"
        )
        
        st.success(f"✅ {filename} ready for download!")
    
    def get_export_summary(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Get summary of exportable data."""
        
        return {
            "total_suppliers": len(analysis_result.suppliers),
            "total_risk_events": len(analysis_result.risk_events),
            "total_ml_predictions": len(analysis_result.ml_predictions),
            "data_quality_score": analysis_result.data_quality_score,
            "analysis_date": analysis_result.generated_at,
            "export_formats_available": self.export_formats,
            "estimated_json_size": f"{len(str(analysis_result.dict())) / 1024:.1f} KB",
            "estimated_csv_size": f"{len(analysis_result.suppliers) * 0.5:.1f} KB"
        }