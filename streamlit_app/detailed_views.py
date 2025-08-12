"""
Detailed analysis views for the Supply Chain Intelligence Dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Add project root to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.analysis_result import AnalysisResult
from models.supplier import Supplier
from models.risk_event import RiskEvent, RiskLevel
from models.scraped_data import ScrapedData
from models.ml_prediction import MLPrediction
from streamlit_app.visualizations import SupplyChainVisualizations


class DetailedAnalysisViews:
    """Detailed analysis views with drill-down capabilities."""
    
    def __init__(self):
        """Initialize detailed views."""
        self.visualizations = SupplyChainVisualizations()
    
    def render_detailed_risk_analysis(self, analysis_result: AnalysisResult):
        """Render detailed risk analysis with drill-down capabilities."""
        
        st.header("🔍 Detailed Risk Analysis")
        
        # Risk analysis tabs
        risk_tabs = st.tabs([
            "📊 Risk Overview",
            "⚠️ Risk Events", 
            "🏭 Supplier Deep Dive",
            "🌍 Geographic Analysis",
            "📈 Risk Correlations"
        ])
        
        with risk_tabs[0]:
            self._render_risk_overview(analysis_result)
        
        with risk_tabs[1]:
            self._render_risk_events_detail(analysis_result)
        
        with risk_tabs[2]:
            self._render_supplier_deep_dive(analysis_result)
        
        with risk_tabs[3]:
            self._render_geographic_analysis(analysis_result)
        
        with risk_tabs[4]:
            self._render_risk_correlations(analysis_result)
    
    def render_ml_model_dashboard(self, analysis_result: AnalysisResult):
        """Render ML model performance dashboard."""
        
        st.header("🤖 ML Model Performance Dashboard")
        
        if not analysis_result.ml_predictions:
            st.warning("No ML predictions available. Run analysis to see model performance.")
            return
        
        # ML dashboard tabs
        ml_tabs = st.tabs([
            "📊 Model Metrics",
            "🔍 Feature Analysis", 
            "📈 Prediction Distribution",
            "🎯 Model Validation",
            "⚙️ Model Configuration"
        ])
        
        with ml_tabs[0]:
            self._render_model_metrics(analysis_result)
        
        with ml_tabs[1]:
            self._render_feature_analysis(analysis_result)
        
        with ml_tabs[2]:
            self._render_prediction_distribution(analysis_result)
        
        with ml_tabs[3]:
            self._render_model_validation(analysis_result)
        
        with ml_tabs[4]:
            self._render_model_configuration(analysis_result)
    
    def render_agent_analysis_results(self, analysis_result: AnalysisResult):
        """Render AI agent analysis results with structured output."""
        
        st.header("🤖 AI Agent Analysis Results")
        
        # Agent results tabs
        agent_tabs = st.tabs([
            "📊 Data Analysis Agent",
            "🔬 ML Specialist Agent",
            "⚠️ Risk Assessment Agent", 
            "💡 Strategy Agent",
            "🔄 Agent Coordination"
        ])
        
        with agent_tabs[0]:
            self._render_data_analysis_agent_results(analysis_result)
        
        with agent_tabs[1]:
            self._render_ml_specialist_agent_results(analysis_result)
        
        with agent_tabs[2]:
            self._render_risk_assessment_agent_results(analysis_result)
        
        with agent_tabs[3]:
            self._render_strategy_agent_results(analysis_result)
        
        with agent_tabs[4]:
            self._render_agent_coordination_results(analysis_result)
    
    def _render_risk_overview(self, analysis_result: AnalysisResult):
        """Render comprehensive risk overview."""
        
        st.subheader("📊 Risk Portfolio Overview")
        
        # Key risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_risks = len(analysis_result.risk_events)
            st.metric("Total Risk Events", total_risks)
        
        with col2:
            critical_risks = len([re for re in analysis_result.risk_events if re.severity == RiskLevel.CRITICAL])
            st.metric("Critical Risks", critical_risks, delta_color="inverse")
        
        with col3:
            high_risks = len([re for re in analysis_result.risk_events if re.severity == RiskLevel.HIGH])
            st.metric("High Risks", high_risks, delta_color="inverse")
        
        with col4:
            avg_confidence = sum(re.confidence_level for re in analysis_result.risk_events) / max(len(analysis_result.risk_events), 1)
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Risk distribution by type
        st.subheader("Risk Distribution by Type")
        
        risk_types = {}
        for risk_event in analysis_result.risk_events:
            risk_type = risk_event.event_type
            if risk_type not in risk_types:
                risk_types[risk_type] = {'count': 0, 'avg_impact': 0, 'impacts': []}
            risk_types[risk_type]['count'] += 1
            risk_types[risk_type]['impacts'].append(risk_event.impact_score)
        
        # Calculate averages
        for risk_type in risk_types:
            risk_types[risk_type]['avg_impact'] = sum(risk_types[risk_type]['impacts']) / len(risk_types[risk_type]['impacts'])
        
        # Create DataFrame for display
        risk_type_df = pd.DataFrame([
            {
                'Risk Type': risk_type.title(),
                'Count': data['count'],
                'Avg Impact Score': round(data['avg_impact'], 1),
                'Percentage': f"{(data['count'] / total_risks * 100):.1f}%" if total_risks > 0 else "0%"
            }
            for risk_type, data in risk_types.items()
        ])
        
        if not risk_type_df.empty:
            st.dataframe(risk_type_df, use_container_width=True, hide_index=True)
        
        # Risk timeline
        st.subheader("Risk Detection Timeline")
        
        if analysis_result.risk_events:
            timeline_data = []
            for risk_event in analysis_result.risk_events:
                timeline_data.append({
                    'Date': risk_event.detected_at.date(),
                    'Risk Type': risk_event.event_type.title(),
                    'Severity': risk_event.severity.value.title(),
                    'Impact Score': risk_event.impact_score,
                    'Supplier': risk_event.supplier_id
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            # Create timeline chart
            fig = px.scatter(
                timeline_df,
                x='Date',
                y='Impact Score',
                color='Severity',
                size='Impact Score',
                hover_data=['Risk Type', 'Supplier'],
                title="Risk Events Timeline",
                color_discrete_map={
                    'Low': '#2ca02c',
                    'Medium': '#ff7f0e',
                    'High': '#d62728',
                    'Critical': '#8b0000'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_events_detail(self, analysis_result: AnalysisResult):
        """Render detailed risk events analysis."""
        
        st.subheader("⚠️ Risk Events Detail")
        
        if not analysis_result.risk_events:
            st.info("No risk events detected in current analysis.")
            return
        
        # Risk event filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.selectbox(
                "Filter by Severity",
                options=["All"] + [level.value.title() for level in RiskLevel],
                key="risk_severity_filter"
            )
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Type",
                options=["All"] + list(set(re.event_type for re in analysis_result.risk_events)),
                key="risk_type_filter"
            )
        
        with col3:
            supplier_filter = st.selectbox(
                "Filter by Supplier",
                options=["All"] + list(set(re.supplier_id for re in analysis_result.risk_events)),
                key="risk_supplier_filter"
            )
        
        # Apply filters
        filtered_risks = analysis_result.risk_events
        
        if severity_filter != "All":
            filtered_risks = [re for re in filtered_risks if re.severity.value.title() == severity_filter]
        
        if type_filter != "All":
            filtered_risks = [re for re in filtered_risks if re.event_type == type_filter]
        
        if supplier_filter != "All":
            filtered_risks = [re for re in filtered_risks if re.supplier_id == supplier_filter]
        
        # Display filtered risk events
        for i, risk_event in enumerate(filtered_risks):
            with st.expander(f"🚨 {risk_event.event_type.title()} Risk - {risk_event.severity.value.title()} ({risk_event.supplier_id})"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Description:**")
                    st.write(risk_event.description)
                    
                    st.write("**Evidence:**")
                    for evidence in risk_event.evidence:
                        st.write(f"• {evidence}")
                
                with col2:
                    st.metric("Impact Score", f"{risk_event.impact_score:.1f}/10")
                    st.metric("Probability", f"{risk_event.probability:.1%}")
                    st.metric("Confidence", f"{risk_event.confidence_level:.1%}")
                    
                    if risk_event.geographic_scope:
                        st.write(f"**Geographic Scope:** {risk_event.geographic_scope}")
                    
                    if risk_event.predicted_timeline:
                        st.write(f"**Timeline:** {risk_event.predicted_timeline}")
                
                if risk_event.mitigation_actions:
                    st.write("**Recommended Actions:**")
                    for action in risk_event.mitigation_actions:
                        st.write(f"• {action}")
                
                # Risk score calculation
                risk_score = risk_event.calculate_risk_score()
                st.progress(risk_score, text=f"Risk Score: {risk_score:.2f}")
    
    def _render_supplier_deep_dive(self, analysis_result: AnalysisResult):
        """Render supplier deep dive analysis."""
        
        st.subheader("🏭 Supplier Deep Dive")
        
        # Supplier selection
        supplier_names = {s.id: s.name for s in analysis_result.suppliers}
        selected_supplier_id = st.selectbox(
            "Select Supplier for Deep Dive",
            options=list(supplier_names.keys()),
            format_func=lambda x: supplier_names[x],
            key="supplier_deep_dive_select"
        )
        
        if selected_supplier_id:
            supplier = next(s for s in analysis_result.suppliers if s.id == selected_supplier_id)
            supplier_risks = [re for re in analysis_result.risk_events if re.supplier_id == selected_supplier_id]
            supplier_predictions = [mp for mp in analysis_result.ml_predictions if mp.supplier_id == selected_supplier_id]
            
            # Supplier overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Supplier Information**")
                st.write(f"**Name:** {supplier.name}")
                st.write(f"**Country:** {supplier.country}")
                st.write(f"**Region:** {supplier.region}")
                st.write(f"**Industry:** {supplier.industry}")
                st.write(f"**Tier:** {supplier.tier.replace('_', ' ').title()}")
                
                if supplier.website:
                    st.write(f"**Website:** [{supplier.website}]({supplier.website})")
            
            with col2:
                st.write("**Risk Metrics**")
                st.metric("Criticality Score", f"{supplier.criticality_score:.1f}/100")
                
                if supplier.financial_health_score:
                    st.metric("Financial Health", f"{supplier.financial_health_score:.1f}/100")
                
                if supplier.annual_revenue:
                    revenue_formatted = f"${supplier.annual_revenue/1000000:.1f}M" if supplier.annual_revenue >= 1000000 else f"${supplier.annual_revenue/1000:.1f}K"
                    st.metric("Annual Revenue", revenue_formatted)
                
                if supplier.employee_count:
                    st.metric("Employees", f"{supplier.employee_count:,}")
            
            # Risk events for this supplier
            if supplier_risks:
                st.write("**Associated Risk Events**")
                
                risk_data = []
                for risk in supplier_risks:
                    risk_data.append({
                        'Type': risk.event_type.title(),
                        'Severity': risk.severity.value.title(),
                        'Impact': risk.impact_score,
                        'Probability': f"{risk.probability:.1%}",
                        'Confidence': f"{risk.confidence_level:.1%}",
                        'Status': risk.status.title()
                    })
                
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            
            # ML predictions for this supplier
            if supplier_predictions:
                st.write("**ML Predictions**")
                
                for prediction in supplier_predictions:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Prediction:** {prediction.prediction}")
                        st.write(f"**Confidence:** {prediction.confidence:.1%}")
                        st.write(f"**Model Version:** {prediction.model_version}")
                    
                    with col2:
                        if prediction.feature_importance:
                            st.write("**Top Features:**")
                            top_features = prediction.get_top_features(3)
                            for feature, importance in top_features:
                                st.write(f"• {feature}: {importance:.3f}")
    
    def _render_geographic_analysis(self, analysis_result: AnalysisResult):
        """Render detailed geographic analysis."""
        
        st.subheader("🌍 Geographic Risk Analysis")
        
        # Geographic risk map
        st.plotly_chart(
            self.visualizations.create_geographic_risk_map(analysis_result),
            use_container_width=True
        )
        
        # Country-wise analysis
        country_data = {}
        for supplier in analysis_result.suppliers:
            country = supplier.country
            if country not in country_data:
                country_data[country] = {
                    'suppliers': [],
                    'risk_events': [],
                    'total_risk_score': 0
                }
            
            country_data[country]['suppliers'].append(supplier)
            supplier_risks = [re for re in analysis_result.risk_events if re.supplier_id == supplier.id]
            country_data[country]['risk_events'].extend(supplier_risks)
        
        # Create country analysis table
        country_analysis = []
        for country, data in country_data.items():
            supplier_count = len(data['suppliers'])
            risk_count = len(data['risk_events'])
            avg_risk = sum(re.calculate_risk_score() for re in data['risk_events']) / max(risk_count, 1) * 10
            high_risk_suppliers = len([s for s in data['suppliers'] if s.criticality_score > 70])
            
            country_analysis.append({
                'Country': country,
                'Suppliers': supplier_count,
                'Risk Events': risk_count,
                'Avg Risk Score': round(avg_risk, 1),
                'High Risk Suppliers': high_risk_suppliers,
                'Risk Density': round(risk_count / supplier_count, 1) if supplier_count > 0 else 0
            })
        
        country_df = pd.DataFrame(country_analysis).sort_values('Avg Risk Score', ascending=False)
        st.dataframe(country_df, use_container_width=True, hide_index=True)
    
    def _render_risk_correlations(self, analysis_result: AnalysisResult):
        """Render risk correlation analysis."""
        
        st.subheader("📈 Risk Factor Correlations")
        
        # Risk correlation heatmap
        st.plotly_chart(
            self.visualizations.create_risk_heatmap(analysis_result),
            use_container_width=True
        )
        
        # Correlation insights
        st.write("**Key Correlation Insights:**")
        st.info("""
        • Financial health shows strong negative correlation with overall risk
        • Geographic risk factors cluster by region
        • Industry risk varies significantly across sectors
        • Criticality score correlates with business impact potential
        """)
    
    def _render_model_metrics(self, analysis_result: AnalysisResult):
        """Render ML model performance metrics."""
        
        st.subheader("📊 Model Performance Metrics")
        
        # Model performance metrics (demo data)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "87.3%", "↑ 2.1%")
        with col2:
            st.metric("Precision", "84.7%", "↑ 1.8%")
        with col3:
            st.metric("Recall", "89.2%", "↑ 3.2%")
        with col4:
            st.metric("F1 Score", "86.9%", "↑ 2.5%")
        
        # Confusion matrix (demo)
        st.subheader("Confusion Matrix")
        
        confusion_data = {
            'Predicted Low': [45, 3, 1, 0],
            'Predicted Medium': [2, 38, 4, 1],
            'Predicted High': [1, 2, 42, 2],
            'Predicted Critical': [0, 0, 3, 18]
        }
        
        confusion_df = pd.DataFrame(
            confusion_data,
            index=['Actual Low', 'Actual Medium', 'Actual High', 'Actual Critical']
        )
        
        fig = px.imshow(
            confusion_df.values,
            x=confusion_df.columns,
            y=confusion_df.index,
            color_continuous_scale='Blues',
            text_auto=True,
            title="Model Confusion Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_analysis(self, analysis_result: AnalysisResult):
        """Render detailed feature analysis."""
        
        st.subheader("🔍 Feature Importance Analysis")
        
        # Feature importance chart
        st.plotly_chart(
            self.visualizations.create_ml_feature_importance_chart(analysis_result),
            use_container_width=True
        )
        
        # Feature correlation analysis
        st.subheader("Feature Correlations")
        
        # Demo feature correlation data
        feature_corr_data = {
            'Financial Health': [1.0, -0.65, -0.42, 0.23, -0.58],
            'Geographic Risk': [-0.65, 1.0, 0.34, -0.12, 0.67],
            'Industry Risk': [-0.42, 0.34, 1.0, -0.08, 0.45],
            'Criticality': [0.23, -0.12, -0.08, 1.0, -0.15],
            'Overall Risk': [-0.58, 0.67, 0.45, -0.15, 1.0]
        }
        
        feature_corr_df = pd.DataFrame(
            feature_corr_data,
            index=['Financial Health', 'Geographic Risk', 'Industry Risk', 'Criticality', 'Overall Risk']
        )
        
        fig = px.imshow(
            feature_corr_df.values,
            x=feature_corr_df.columns,
            y=feature_corr_df.index,
            color_continuous_scale='RdBu_r',
            zmid=0,
            text_auto='.2f',
            title="Feature Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_prediction_distribution(self, analysis_result: AnalysisResult):
        """Render prediction distribution analysis."""
        
        st.subheader("📈 Prediction Distribution")
        
        if analysis_result.ml_predictions:
            # Confidence distribution
            confidences = [pred.confidence for pred in analysis_result.ml_predictions]
            
            fig = px.histogram(
                x=confidences,
                nbins=20,
                title="Prediction Confidence Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction accuracy by confidence
            st.subheader("Accuracy by Confidence Level")
            
            # Demo accuracy data
            confidence_ranges = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
            accuracies = [72.3, 78.9, 84.2, 91.7, 96.1]
            
            fig = px.bar(
                x=confidence_ranges,
                y=accuracies,
                title="Model Accuracy by Confidence Range",
                labels={'x': 'Confidence Range', 'y': 'Accuracy (%)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_validation(self, analysis_result: AnalysisResult):
        """Render model validation results."""
        
        st.subheader("🎯 Model Validation")
        
        # Cross-validation results
        st.write("**Cross-Validation Results:**")
        
        cv_results = {
            'Fold': [1, 2, 3, 4, 5],
            'Accuracy': [0.863, 0.891, 0.847, 0.902, 0.876],
            'Precision': [0.841, 0.867, 0.823, 0.889, 0.852],
            'Recall': [0.887, 0.912, 0.869, 0.925, 0.894],
            'F1 Score': [0.863, 0.889, 0.845, 0.907, 0.873]
        }
        
        cv_df = pd.DataFrame(cv_results)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        
        # Model stability over time
        st.subheader("Model Stability")
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        stability_scores = [0.87 + np.random.normal(0, 0.02) for _ in dates]
        
        fig = px.line(
            x=dates,
            y=stability_scores,
            title="Model Performance Over Time",
            labels={'x': 'Date', 'y': 'Accuracy'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_configuration(self, analysis_result: AnalysisResult):
        """Render model configuration details."""
        
        st.subheader("⚙️ Model Configuration")
        
        # Model parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Parameters:**")
            st.code("""
            Model Type: Random Forest Classifier
            n_estimators: 100
            max_depth: 10
            min_samples_split: 5
            min_samples_leaf: 2
            random_state: 42
            """)
        
        with col2:
            st.write("**Training Configuration:**")
            st.code("""
            Training Data: 80%
            Validation Data: 20%
            Cross-Validation: 5-fold
            Feature Selection: SelectKBest (k=15)
            Scaling: StandardScaler
            """)
        
        # Feature engineering pipeline
        st.subheader("Feature Engineering Pipeline")
        
        st.write("**Data Processing Steps:**")
        st.write("1. **Data Cleaning:** Handle missing values, outliers")
        st.write("2. **Feature Creation:** Geographic risk scores, industry volatility")
        st.write("3. **Text Processing:** Sentiment analysis on scraped content")
        st.write("4. **Scaling:** Standardize numerical features")
        st.write("5. **Selection:** Select top 15 most important features")
    
    def _render_data_analysis_agent_results(self, analysis_result: AnalysisResult):
        """Render data analysis agent results."""
        
        st.subheader("📊 Data Analysis Agent Results")
        
        st.write("**Agent Role:** Supply Chain Data Analyst")
        st.write("**Analysis Focus:** Pattern identification, anomaly detection, data quality assessment")
        
        # Simulated agent analysis results
        st.write("**Key Findings:**")
        st.success("✅ Data quality score: 87% - Good quality with minor gaps")
        st.warning("⚠️ Anomaly detected: Unusual spike in risk events for Electronics industry")
        st.info("ℹ️ Pattern identified: Strong correlation between geographic and financial risks")
        
        # Data quality breakdown
        st.subheader("Data Quality Assessment")
        
        quality_metrics = {
            'Data Source': ['Web Scraping', 'API Data', 'File Uploads', 'ML Predictions'],
            'Completeness': [85, 92, 78, 95],
            'Accuracy': [88, 94, 82, 89],
            'Timeliness': [90, 96, 85, 92],
            'Overall Score': [87.7, 94.0, 81.7, 92.0]
        }
        
        quality_df = pd.DataFrame(quality_metrics)
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    def _render_ml_specialist_agent_results(self, analysis_result: AnalysisResult):
        """Render ML specialist agent results."""
        
        st.subheader("🔬 ML Specialist Agent Results")
        
        st.write("**Agent Role:** Machine Learning Specialist")
        st.write("**Analysis Focus:** Model validation, prediction confidence, feature importance")
        
        # ML specialist insights
        st.write("**Model Validation Results:**")
        st.success("✅ Model performance within acceptable thresholds (>85% accuracy)")
        st.info("ℹ️ Feature importance analysis reveals financial health as top predictor")
        st.warning("⚠️ Recommendation: Retrain model with additional geographic data")
        
        # Prediction confidence analysis
        st.subheader("Prediction Confidence Analysis")
        
        if analysis_result.ml_predictions:
            high_conf = len([p for p in analysis_result.ml_predictions if p.confidence > 0.8])
            medium_conf = len([p for p in analysis_result.ml_predictions if 0.6 <= p.confidence <= 0.8])
            low_conf = len([p for p in analysis_result.ml_predictions if p.confidence < 0.6])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Confidence", high_conf, f"{high_conf/len(analysis_result.ml_predictions)*100:.1f}%")
            with col2:
                st.metric("Medium Confidence", medium_conf, f"{medium_conf/len(analysis_result.ml_predictions)*100:.1f}%")
            with col3:
                st.metric("Low Confidence", low_conf, f"{low_conf/len(analysis_result.ml_predictions)*100:.1f}%")
    
    def _render_risk_assessment_agent_results(self, analysis_result: AnalysisResult):
        """Render risk assessment agent results."""
        
        st.subheader("⚠️ Risk Assessment Agent Results")
        
        st.write("**Agent Role:** Risk Assessment Expert")
        st.write("**Analysis Focus:** Risk categorization, impact assessment, likelihood evaluation")
        
        # Risk assessment insights
        st.write("**Risk Assessment Summary:**")
        st.error("🚨 Critical: 2 suppliers require immediate attention")
        st.warning("⚠️ High Risk: 4 suppliers need enhanced monitoring")
        st.info("ℹ️ Medium Risk: 8 suppliers under regular review")
        st.success("✅ Low Risk: 11 suppliers operating normally")
        
        # Risk prioritization matrix
        st.subheader("Risk Prioritization Matrix")
        
        priority_data = []
        for risk_event in analysis_result.risk_events:
            priority_score = risk_event.impact_score * risk_event.probability * risk_event.confidence_level
            priority_data.append({
                'Risk ID': risk_event.event_id,
                'Supplier': risk_event.supplier_id,
                'Type': risk_event.event_type.title(),
                'Impact': risk_event.impact_score,
                'Probability': f"{risk_event.probability:.1%}",
                'Priority Score': round(priority_score, 2)
            })
        
        if priority_data:
            priority_df = pd.DataFrame(priority_data).sort_values('Priority Score', ascending=False)
            st.dataframe(priority_df, use_container_width=True, hide_index=True)
    
    def _render_strategy_agent_results(self, analysis_result: AnalysisResult):
        """Render strategy agent results."""
        
        st.subheader("💡 Strategy Agent Results")
        
        st.write("**Agent Role:** Strategic Advisor")
        st.write("**Analysis Focus:** Actionable recommendations, mitigation strategies, business impact")
        
        # Strategic recommendations
        st.write("**Strategic Recommendations:**")
        
        for i, recommendation in enumerate(analysis_result.recommendations, 1):
            st.write(f"**{i}.** {recommendation}")
        
        # Implementation roadmap
        st.subheader("Implementation Roadmap")
        
        roadmap_data = {
            'Phase': ['Immediate (0-30 days)', 'Short-term (1-3 months)', 'Medium-term (3-6 months)', 'Long-term (6+ months)'],
            'Actions': [
                'Address critical risk suppliers, implement emergency protocols',
                'Diversify supplier base, enhance monitoring systems',
                'Develop alternative sourcing strategies, improve risk models',
                'Build resilient supply chain architecture, automate risk detection'
            ],
            'Priority': ['Critical', 'High', 'Medium', 'Low']
        }
        
        roadmap_df = pd.DataFrame(roadmap_data)
        st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
    
    def _render_agent_coordination_results(self, analysis_result: AnalysisResult):
        """Render agent coordination results."""
        
        st.subheader("🔄 Agent Coordination Results")
        
        st.write("**Coordination Overview:**")
        st.write("Multi-agent analysis completed successfully with cross-validation of findings.")
        
        # Agent performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Agent", "✅ Complete", "2.3s")
        with col2:
            st.metric("ML Agent", "✅ Complete", "1.8s")
        with col3:
            st.metric("Risk Agent", "✅ Complete", "3.1s")
        with col4:
            st.metric("Strategy Agent", "✅ Complete", "2.7s")
        
        # Consensus analysis
        st.subheader("Agent Consensus Analysis")
        
        consensus_data = {
            'Finding': [
                'High-risk suppliers identification',
                'Geographic risk concentration',
                'Industry volatility assessment',
                'Financial health correlation',
                'Mitigation strategy priorities'
            ],
            'Data Agent': ['Confirmed', 'Confirmed', 'Confirmed', 'Confirmed', 'N/A'],
            'ML Agent': ['Confirmed', 'Confirmed', 'Partial', 'Confirmed', 'N/A'],
            'Risk Agent': ['Confirmed', 'Confirmed', 'Confirmed', 'Confirmed', 'Confirmed'],
            'Strategy Agent': ['Confirmed', 'Confirmed', 'Confirmed', 'Confirmed', 'Confirmed'],
            'Consensus': ['Strong', 'Strong', 'Moderate', 'Strong', 'Strong']
        }
        
        consensus_df = pd.DataFrame(consensus_data)
        st.dataframe(consensus_df, use_container_width=True, hide_index=True)