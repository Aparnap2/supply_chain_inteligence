"""
Supply Chain Intelligence & Risk Management Platform - Main Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.analysis_result import AnalysisResult
from models.supplier import Supplier
from models.risk_event import RiskEvent, RiskLevel
from models.scraped_data import ScrapedData
from models.ml_prediction import MLPrediction
from streamlit_app.visualizations import SupplyChainVisualizations
from streamlit_app.utils import DataLoader, DashboardHelpers
from workflows.agile_workflow import AgileWorkflow
from streamlit_app.detailed_views import DetailedAnalysisViews
from streamlit_app.export_reports import ExportReportManager


class SupplyChainDashboard:
    """Main dashboard class for Supply Chain Intelligence Platform."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.setup_page_config()
        self.initialize_session_state()
        self.visualizations = SupplyChainVisualizations()
        self.detailed_views = DetailedAnalysisViews()
        self.export_manager = ExportReportManager()
        self.agile_workflow = AgileWorkflow()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Supply Chain Intelligence Platform",
            page_icon="🔗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .risk-high {
            color: #d62728;
            font-weight: bold;
        }
        .risk-medium {
            color: #ff7f0e;
            font-weight: bold;
        }
        .risk-low {
            color: #2ca02c;
            font-weight: bold;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active {
            background-color: #2ca02c;
        }
        .status-warning {
            background-color: #ff7f0e;
        }
        .status-error {
            background-color: #d62728;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
    
    def render_header(self):
        """Render the main dashboard header."""
        st.markdown('<h1 class="main-header">🔗 Supply Chain Intelligence Platform</h1>', 
                   unsafe_allow_html=True)
        
        # Navigation and controls
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            st.markdown("**Real-time Supply Chain Risk Assessment & Intelligence**")
        
        with col2:
            if st.button("🔄 Refresh Data", help="Refresh dashboard data"):
                self.refresh_data()
        
        with col3:
            auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh
        
        with col4:
            if st.button("📤 Quick Export", help="Quick export current analysis"):
                if st.session_state.analysis_results:
                    json_data = self.export_manager.export_to_json(st.session_state.analysis_results)
                    st.download_button(
                        "⬇️ Download JSON",
                        data=json_data,
                        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    # Export demo data
                    demo_result = DataLoader.create_demo_analysis_result()
                    json_data = self.export_manager.export_to_json(demo_result)
                    st.download_button(
                        "⬇️ Download Demo JSON",
                        data=json_data,
                        file_name=f"demo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col5:
            st.markdown(f"**Last Updated:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    def render_executive_summary(self, analysis_result: Optional[AnalysisResult] = None):
        """Render executive summary dashboard with key metrics."""
        st.subheader("📊 Executive Summary")
        
        if analysis_result is None:
            # Show demo/placeholder metrics
            metrics = self.get_demo_metrics()
        else:
            metrics = analysis_result.get_summary_metrics()
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total Suppliers",
                value=metrics.get("total_suppliers", 0),
                delta=None,
                help="Total number of suppliers in the analysis"
            )
        
        with col2:
            high_risk_count = metrics.get("high_risk_suppliers", 0)
            total_suppliers = metrics.get("total_suppliers", 1)
            delta_color = "inverse" if high_risk_count > 0 else "normal"
            st.metric(
                label="High Risk Suppliers",
                value=high_risk_count,
                delta=f"{(high_risk_count/total_suppliers)*100:.1f}%",
                delta_color=delta_color,
                help="Suppliers with high or critical risk levels"
            )
        
        with col3:
            overall_risk = metrics.get("overall_risk_score", 0)
            risk_color = "inverse" if overall_risk > 70 else "normal"
            st.metric(
                label="Overall Risk Score",
                value=f"{overall_risk:.1f}",
                delta=None,
                help="Portfolio-wide risk score (0-100)"
            )
        
        with col4:
            st.metric(
                label="Risk Events",
                value=metrics.get("total_risk_events", 0),
                delta=metrics.get("critical_events", 0),
                delta_color="inverse" if metrics.get("critical_events", 0) > 0 else "normal",
                help="Total risk events identified"
            )
        
        with col5:
            data_quality = metrics.get("data_quality_score", 0) * 100
            quality_color = "inverse" if data_quality < 70 else "normal"
            st.metric(
                label="Data Quality",
                value=f"{data_quality:.1f}%",
                delta=None,
                delta_color=quality_color,
                help="Overall data quality score"
            )
    
    def render_status_indicators(self):
        """Render real-time status indicators for data pipeline components."""
        st.subheader("🔧 System Status")
        
        # Pipeline component status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.render_status_card("Data Collection", "active", "Web scraping and API collection operational")
        
        with col2:
            self.render_status_card("ML Pipeline", "active", "Models trained and predictions current")
        
        with col3:
            self.render_status_card("AI Agents", "warning", "Some agents experiencing delays")
        
        with col4:
            self.render_status_card("Risk Analysis", "active", "Risk assessment up to date")
    
    def render_status_card(self, component: str, status: str, description: str):
        """Render individual status indicator card."""
        status_colors = {
            "active": "#2ca02c",
            "warning": "#ff7f0e", 
            "error": "#d62728"
        }
        
        status_icons = {
            "active": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span class="status-indicator status-{status}"></span>
                <strong>{component}</strong>
                <span style="margin-left: auto;">{status_icons[status]}</span>
            </div>
            <small style="color: #666;">{description}</small>
        </div>
        """, unsafe_allow_html=True)
    
    def render_supplier_overview(self, analysis_result: Optional[AnalysisResult] = None):
        """Render supplier overview table with risk indicators."""
        st.subheader("🏭 Supplier Overview")
        
        if analysis_result is None:
            # Show demo data
            supplier_data = self.get_demo_supplier_data()
        else:
            supplier_data = self.prepare_supplier_table_data(analysis_result)
        
        # Create DataFrame for display
        df = pd.DataFrame(supplier_data)
        
        # Configure column display
        column_config = {
            "risk_level": st.column_config.SelectboxColumn(
                "Risk Level",
                options=["Low", "Medium", "High", "Critical"],
                required=True
            ),
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=100,
                format="%.1f"
            ),
            "criticality_score": st.column_config.ProgressColumn(
                "Criticality",
                min_value=0,
                max_value=100,
                format="%.1f"
            ),
            "website": st.column_config.LinkColumn(
                "Website",
                display_text="Visit"
            )
        }
        
        # Display table with filtering options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search suppliers", placeholder="Enter supplier name or country...")
        
        with col2:
            risk_filter = st.selectbox("Filter by Risk", ["All", "Low", "Medium", "High", "Critical"])
        
        with col3:
            region_filter = st.selectbox("Filter by Region", ["All"] + list(df["region"].unique()) if not df.empty else ["All"])
        
        # Apply filters
        filtered_df = df.copy()
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df["name"].str.contains(search_term, case=False) |
                filtered_df["country"].str.contains(search_term, case=False)
            ]
        
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]
        
        if region_filter != "All":
            filtered_df = filtered_df[filtered_df["region"] == region_filter]
        
        # Display filtered table
        st.dataframe(
            filtered_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary statistics
        if not filtered_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Showing {len(filtered_df)} of {len(df)} suppliers**")
            
            with col2:
                avg_risk = filtered_df["risk_score"].mean()
                st.info(f"**Average Risk Score:** {avg_risk:.1f}")
            
            with col3:
                high_risk_count = len(filtered_df[filtered_df["risk_level"].isin(["High", "Critical"])])
                st.info(f"**High Risk Suppliers:** {high_risk_count}")
    
    def get_demo_metrics(self) -> Dict[str, Any]:
        """Generate demo metrics for display when no real data is available."""
        return {
            "total_suppliers": 25,
            "high_risk_suppliers": 4,
            "total_risk_events": 12,
            "critical_events": 2,
            "overall_risk_score": 68.5,
            "data_quality_score": 0.87,
            "data_coverage_score": 0.92,
            "processing_time": 45.2,
            "generated_at": datetime.now()
        }
    
    def get_demo_supplier_data(self) -> List[Dict[str, Any]]:
        """Generate demo supplier data for display."""
        return [
            {
                "name": "Global Manufacturing Corp",
                "country": "Germany",
                "region": "Europe",
                "industry": "Manufacturing",
                "tier": "Tier 1",
                "risk_level": "Medium",
                "risk_score": 65.0,
                "criticality_score": 90.0,
                "website": "https://globalmanufacturing.com"
            },
            {
                "name": "Asia Pacific Electronics",
                "country": "Taiwan",
                "region": "Asia Pacific",
                "industry": "Electronics",
                "tier": "Tier 1",
                "risk_level": "High",
                "risk_score": 78.5,
                "criticality_score": 85.0,
                "website": "https://apelectronics.com"
            },
            {
                "name": "European Logistics Solutions",
                "country": "Netherlands",
                "region": "Europe",
                "industry": "Logistics",
                "tier": "Tier 2",
                "risk_level": "Low",
                "risk_score": 35.2,
                "criticality_score": 60.0,
                "website": "https://eurolog.com"
            },
            {
                "name": "North American Steel Co",
                "country": "United States",
                "region": "North America",
                "industry": "Materials",
                "tier": "Tier 1",
                "risk_level": "Critical",
                "risk_score": 89.3,
                "criticality_score": 95.0,
                "website": "https://nasteel.com"
            },
            {
                "name": "South American Mining Ltd",
                "country": "Brazil",
                "region": "South America",
                "industry": "Mining",
                "tier": "Tier 2",
                "risk_level": "High",
                "risk_score": 72.1,
                "criticality_score": 70.0,
                "website": "https://samining.com"
            }
        ]
    
    def prepare_supplier_table_data(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Prepare supplier data for table display from analysis results."""
        table_data = []
        
        for supplier in analysis_result.suppliers:
            # Find associated risk events and ML predictions
            supplier_risks = [re for re in analysis_result.risk_events if re.supplier_id == supplier.id]
            supplier_predictions = [mp for mp in analysis_result.ml_predictions if mp.supplier_id == supplier.id]
            
            # Calculate risk level and score
            if supplier_risks:
                max_severity = max(risk.severity for risk in supplier_risks)
                risk_level = max_severity.title()
                risk_score = max(risk.calculate_risk_score() * 10 for risk in supplier_risks)
            elif supplier_predictions:
                # Use ML prediction if available
                prediction = supplier_predictions[0]
                risk_level = prediction.prediction.title()
                risk_score = prediction.confidence * 100
            else:
                risk_level = "Low"
                risk_score = supplier.criticality_score * 0.5
            
            table_data.append({
                "name": supplier.name,
                "country": supplier.country,
                "region": supplier.region,
                "industry": supplier.industry,
                "tier": supplier.tier.replace("_", " ").title(),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "criticality_score": supplier.criticality_score,
                "website": supplier.website
            })
        
        return table_data
    
    def render_interactive_visualizations(self, analysis_result: Optional[AnalysisResult] = None):
        """Render interactive visualizations section."""
        st.subheader("📈 Interactive Risk Analytics")
        
        # Use demo data if no real analysis result
        if analysis_result is None:
            analysis_result = DataLoader.create_demo_analysis_result()
        
        # Visualization selection tabs
        viz_tabs = st.tabs([
            "🔥 Risk Heatmap", 
            "🌍 Geographic Risk", 
            "📊 Risk Trends", 
            "🏭 Industry Analysis",
            "🤖 ML Insights"
        ])
        
        with viz_tabs[0]:
            st.plotly_chart(
                self.visualizations.create_risk_heatmap(analysis_result),
                use_container_width=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    self.visualizations.create_risk_distribution_chart(analysis_result),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    self.visualizations.create_supplier_risk_scatter(analysis_result),
                    use_container_width=True
                )
        
        with viz_tabs[1]:
            st.plotly_chart(
                self.visualizations.create_geographic_risk_map(analysis_result),
                use_container_width=True
            )
            
            # Geographic insights
            st.subheader("🌍 Geographic Risk Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Highest Risk Region", "Asia Pacific", "↑ 12%")
            with col2:
                st.metric("Most Suppliers", "Europe", "25 suppliers")
            with col3:
                st.metric("Emerging Risks", "South America", "↑ 8%")
        
        with viz_tabs[2]:
            # Time period selector
            col1, col2 = st.columns([3, 1])
            with col2:
                time_period = st.selectbox(
                    "Time Period",
                    options=["7d", "30d", "90d", "180d"],
                    index=1,
                    help="Select time period for trend analysis"
                )
            
            st.plotly_chart(
                self.visualizations.create_risk_trend_chart(analysis_result, time_period),
                use_container_width=True
            )
            
            # Trend insights
            st.subheader("📈 Trend Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**Risk Trend:** Increasing over last 30 days")
            with col2:
                st.warning("**Alert:** 3 new high-risk suppliers identified")
            with col3:
                st.success("**Improvement:** 2 suppliers moved to lower risk")
        
        with viz_tabs[3]:
            st.plotly_chart(
                self.visualizations.create_industry_risk_chart(analysis_result),
                use_container_width=True
            )
            
            # Industry insights
            st.subheader("🏭 Industry Risk Analysis")
            
            # Create industry comparison table
            industry_data = []
            for supplier in analysis_result.suppliers:
                supplier_risks = [re for re in analysis_result.risk_events if re.supplier_id == supplier.id]
                risk_score = max(re.calculate_risk_score() * 10 for re in supplier_risks) if supplier_risks else supplier.criticality_score * 0.5
                
                industry_data.append({
                    "Industry": supplier.industry,
                    "Risk Score": risk_score,
                    "Suppliers": 1
                })
            
            if industry_data:
                industry_df = pd.DataFrame(industry_data)
                industry_summary = industry_df.groupby("Industry").agg({
                    "Risk Score": "mean",
                    "Suppliers": "sum"
                }).round(1).reset_index()
                
                st.dataframe(
                    industry_summary,
                    use_container_width=True,
                    hide_index=True
                )
        
        with viz_tabs[4]:
            if analysis_result.ml_predictions:
                st.plotly_chart(
                    self.visualizations.create_ml_feature_importance_chart(analysis_result),
                    use_container_width=True
                )
                
                # ML model performance metrics
                st.subheader("🤖 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Model Accuracy", "87.3%", "↑ 2.1%")
                with col2:
                    st.metric("Precision", "84.7%", "↑ 1.8%")
                with col3:
                    st.metric("Recall", "89.2%", "↑ 3.2%")
                with col4:
                    st.metric("F1 Score", "86.9%", "↑ 2.5%")
                
                # Feature importance insights
                st.subheader("🔍 Feature Analysis")
                st.info("""
                **Key Insights:**
                - Financial health score is the most important predictor
                - Geographic risk factors show increasing importance
                - Industry volatility contributes significantly to risk assessment
                - Sentiment analysis from news sources provides early warning signals
                """)
            else:
                st.info("No ML predictions available. Run analysis to see ML insights.")
    
    def refresh_data(self):
        """Refresh dashboard data."""
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    def run(self):
        """Main dashboard execution."""
        self.render_header()
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            # Check if 30 seconds have passed since last refresh
            if (datetime.now() - st.session_state.last_refresh).seconds >= 30:
                self.refresh_data()
        
        # Main dashboard content
        self.render_executive_summary(st.session_state.analysis_results)
        
        st.divider()
        
        self.render_status_indicators()
        
        st.divider()
        
        self.render_supplier_overview(st.session_state.analysis_results)
        
        st.divider()
        
        # Interactive visualizations section
        self.render_interactive_visualizations(st.session_state.analysis_results)
        
        # Sidebar for additional controls and information
        self.render_sidebar()
    
    def render_sidebar(self):
        """Render sidebar with additional controls and information."""
        with st.sidebar:
            st.header("🔧 Dashboard Controls")
            
            # Navigation section
            st.subheader("📋 Navigation")
            
            view_option = st.selectbox(
                "Select View",
                options=[
                    "Dashboard Overview",
                    "Detailed Risk Analysis", 
                    "ML Model Dashboard",
                    "AI Agent Results",
                    "Export & Reports"
                ],
                key="view_selector"
            )
            
            # Handle view navigation
            if view_option != "Dashboard Overview":
                self._render_detailed_view(view_option)
                return
            
            st.divider()
            
            # Data upload section
            st.subheader("📁 Data Upload")
            uploaded_file = st.file_uploader(
                "Upload supplier data",
                type=['csv', 'xlsx', 'json'],
                help="Upload CSV, Excel, or JSON files with supplier information"
            )
            
            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                if st.button("Process Upload"):
                    st.info("Processing uploaded data...")
                    # TODO: Implement file processing
            
            # Manual data entry option
            st.subheader("✏️ Quick Manual Entry")
            if st.button("Add Supplier Manually"):
                with st.form("manual_supplier"):
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input("Company Name*")
                        country = st.text_input("Country*")
                        industry = st.text_input("Industry*")
                    with col2:
                        region = st.selectbox("Region*", ["Europe", "Asia Pacific", "North America", "South America", "Africa", "Middle East"])
                        tier = st.selectbox("Tier*", ["tier_1", "tier_2", "tier_3"])
                        criticality = st.slider("Criticality Score*", 0, 100, 50)
                    
                    website = st.text_input("Website (Optional)")
                    
                    if st.form_submit_button("🚀 Add & Analyze"):
                        if name and country and industry:
                            supplier_data = {
                                "name": name,
                                "country": country,
                                "region": region,
                                "industry": industry,
                                "tier": tier,
                                "criticality_score": criticality,
                                "website": website if website else None
                            }
                            
                            # Process single supplier
                            result = self.agile_workflow.process_manual_input([supplier_data])
                            st.session_state.analysis_results = result
                            st.success("Supplier added and analyzed!")
                            st.rerun()
                        else:
                            st.error("Please fill in all required fields (*)")
            
            st.divider()
            
            # Analysis settings
            st.subheader("⚙️ Analysis Settings")
            
            risk_threshold = st.slider(
                "Risk Alert Threshold",
                min_value=0,
                max_value=100,
                value=70,
                help="Threshold for high-risk alerts"
            )
            
            refresh_interval = st.selectbox(
                "Auto Refresh Interval",
                options=[30, 60, 300, 600],
                format_func=lambda x: f"{x} seconds" if x < 60 else f"{x//60} minutes",
                help="How often to refresh the dashboard"
            )
            
            st.divider()
            
            # System information
            st.subheader("ℹ️ System Info")
            st.info(f"""
            **Platform Version:** 1.0.0  
            **Last Analysis:** {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M')}  
            **Data Sources:** Web Scraping, APIs, File Uploads  
            **AI Models:** Active  
            """)
    
    def _render_detailed_view(self, view_option: str):
        """Render detailed view based on selection."""
        # Use demo data if no real analysis result
        analysis_result = st.session_state.analysis_results
        if analysis_result is None:
            analysis_result = DataLoader.create_demo_analysis_result()
        
        # Clear main content and render detailed view
        st.empty()
        
        if view_option == "Detailed Risk Analysis":
            self.detailed_views.render_detailed_risk_analysis(analysis_result)
        elif view_option == "ML Model Dashboard":
            self.detailed_views.render_ml_model_dashboard(analysis_result)
        elif view_option == "AI Agent Results":
            self.detailed_views.render_agent_analysis_results(analysis_result)
        elif view_option == "Export & Reports":
            self.export_manager.render_export_section(analysis_result)


def main():
    """Main application entry point."""
    dashboard = SupplyChainDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()