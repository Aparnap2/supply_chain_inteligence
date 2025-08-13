"""
Interactive visualizations for the Supply Chain Intelligence Dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

# Add project root to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.analysis_result import AnalysisResult
from models.supplier import Supplier
from models.risk_event import RiskEvent, RiskLevel
from models.ml_prediction import MLPrediction


class SupplyChainVisualizations:
    """Interactive visualizations for supply chain risk analysis."""
    
    def __init__(self):
        """Initialize visualization settings."""
        self.color_palette = {
            'low': '#2ca02c',      # Green
            'medium': '#ff7f0e',   # Orange  
            'high': '#d62728',     # Red
            'critical': '#8b0000', # Dark Red
            'primary': '#1f77b4',  # Blue
            'secondary': '#ff7f0e' # Orange
        }
        
        self.plotly_theme = "plotly_white"
        self.default_height = 400
    
    def create_risk_heatmap(self, analysis_result: AnalysisResult) -> go.Figure:
        """Create risk correlation heatmap using Plotly."""
        
        # Prepare data for heatmap
        risk_data = self._prepare_risk_correlation_data(analysis_result)
        
        if risk_data.empty:
            return self._create_empty_chart("No risk correlation data available")
        
        # Create correlation matrix
        correlation_matrix = risk_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdYlBu_r',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Risk Factor Correlation Matrix',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Risk Factors",
            yaxis_title="Risk Factors",
            height=self.default_height,
            template=self.plotly_theme
        )
        
        return fig
    
    def create_geographic_risk_map(self, analysis_result: AnalysisResult) -> go.Figure:
        """Create geographic risk distribution map."""
        
        # Prepare geographic data
        geo_data = self._prepare_geographic_data(analysis_result)
        
        if geo_data.empty:
            return self._create_empty_chart("No geographic data available")
        
        # Create choropleth map
        fig = go.Figure(data=go.Choropleth(
            locations=geo_data['country_code'],
            z=geo_data['avg_risk_score'],
            text=geo_data['country'],
            colorscale='Reds',
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_title="Average Risk Score",
            hovertemplate='<b>%{text}</b><br>' +
                         'Risk Score: %{z:.1f}<br>' +
                         'Suppliers: %{customdata[0]}<br>' +
                         'High Risk: %{customdata[1]}<extra></extra>',
            customdata=geo_data[['supplier_count', 'high_risk_count']].values
        ))
        
        fig.update_layout(
            title={
                'text': 'Global Supply Chain Risk Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=self.default_height,
            template=self.plotly_theme
        )
        
        return fig
    
    def create_risk_trend_chart(self, analysis_result: AnalysisResult, 
                               time_period: str = "30d") -> go.Figure:
        """Create time series chart for risk trend analysis."""
        
        # Generate time series data (demo data for now)
        trend_data = self._generate_risk_trend_data(analysis_result, time_period)
        
        fig = go.Figure()
        
        # Add overall risk trend
        fig.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data['overall_risk'],
            mode='lines+markers',
            name='Overall Risk Score',
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>Overall Risk</b><br>Date: %{x}<br>Score: %{y:.1f}<extra></extra>'
        ))
        
        # Add high-risk supplier count
        fig.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data['high_risk_suppliers'],
            mode='lines+markers',
            name='High Risk Suppliers',
            line=dict(color=self.color_palette['high'], width=2),
            marker=dict(size=5),
            yaxis='y2',
            hovertemplate='<b>High Risk Suppliers</b><br>Date: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add risk events
        fig.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data['risk_events'],
            mode='lines+markers',
            name='Risk Events',
            line=dict(color=self.color_palette['secondary'], width=2),
            marker=dict(size=5),
            yaxis='y2',
            hovertemplate='<b>Risk Events</b><br>Date: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'Risk Trends - Last {time_period}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Date",
            yaxis=dict(
                title="Risk Score",
                side="left",
                range=[0, 100]
            ),
            yaxis2=dict(
                title="Count",
                side="right",
                overlaying="y",
                range=[0, max(trend_data['high_risk_suppliers'].max(), 
                              trend_data['risk_events'].max()) * 1.1]
            ),
            height=self.default_height,
            template=self.plotly_theme,
            hovermode='x unified'
        )
        
        return fig
    
    def create_risk_distribution_chart(self, analysis_result: AnalysisResult) -> go.Figure:
        """Create risk level distribution chart."""
        
        risk_distribution = analysis_result.get_risk_distribution()
        
        # Prepare data
        risk_levels = list(risk_distribution.keys())
        counts = list(risk_distribution.values())
        colors = [self.color_palette.get(level, '#gray') for level in risk_levels]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[level.title() for level in risk_levels],
            values=counts,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent+value',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'Risk Level Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=self.default_height,
            template=self.plotly_theme,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_supplier_risk_scatter(self, analysis_result: AnalysisResult) -> go.Figure:
        """Create scatter plot of supplier risk vs criticality."""
        
        scatter_data = self._prepare_supplier_scatter_data(analysis_result)
        
        if scatter_data.empty:
            return self._create_empty_chart("No supplier data available")
        
        fig = go.Figure()
        
        # Group by risk level for different colors
        for risk_level in ['low', 'medium', 'high', 'critical']:
            level_data = scatter_data[scatter_data['risk_level'] == risk_level]
            if not level_data.empty:
                fig.add_trace(go.Scatter(
                    x=level_data['criticality_score'],
                    y=level_data['risk_score'],
                    mode='markers',
                    name=risk_level.title(),
                    marker=dict(
                        color=self.color_palette.get(risk_level, '#gray'),
                        size=level_data['size'],
                        sizemode='diameter',
                        sizeref=2.*max(level_data['size'])/(40.**2),
                        sizemin=4,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    text=level_data['name'],
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Risk Score: %{y:.1f}<br>' +
                                 'Criticality: %{x:.1f}<br>' +
                                 'Country: %{customdata[0]}<br>' +
                                 'Industry: %{customdata[1]}<extra></extra>',
                    customdata=level_data[['country', 'industry']].values
                ))
        
        fig.update_layout(
            title={
                'text': 'Supplier Risk vs Business Criticality',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Business Criticality Score",
            yaxis_title="Risk Score",
            height=self.default_height,
            template=self.plotly_theme,
            showlegend=True
        )
        
        # Add quadrant lines
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant annotations
        fig.add_annotation(x=25, y=75, text="Low Priority<br>High Risk", 
                          showarrow=False, font=dict(size=10, color="gray"))
        fig.add_annotation(x=75, y=75, text="High Priority<br>High Risk", 
                          showarrow=False, font=dict(size=10, color="red"))
        fig.add_annotation(x=25, y=25, text="Low Priority<br>Low Risk", 
                          showarrow=False, font=dict(size=10, color="gray"))
        fig.add_annotation(x=75, y=25, text="High Priority<br>Low Risk", 
                          showarrow=False, font=dict(size=10, color="green"))
        
        return fig
    
    def create_industry_risk_chart(self, analysis_result: AnalysisResult) -> go.Figure:
        """Create industry risk comparison chart."""
        
        industry_data = self._prepare_industry_data(analysis_result)
        
        if industry_data.empty:
            return self._create_empty_chart("No industry data available")
        
        fig = go.Figure()
        
        # Add bar chart for average risk score
        fig.add_trace(go.Bar(
            x=industry_data['industry'],
            y=industry_data['avg_risk_score'],
            name='Average Risk Score',
            marker_color=self.color_palette['primary'],
            text=industry_data['avg_risk_score'].round(1),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Avg Risk Score: %{y:.1f}<br>' +
                         'Suppliers: %{customdata[0]}<br>' +
                         'High Risk: %{customdata[1]}<extra></extra>',
            customdata=industry_data[['supplier_count', 'high_risk_count']].values
        ))
        
        fig.update_layout(
            title={
                'text': 'Risk Analysis by Industry',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Industry",
            yaxis_title="Average Risk Score",
            height=self.default_height,
            template=self.plotly_theme,
            showlegend=False
        )
        
        return fig
    
    def create_ml_feature_importance_chart(self, analysis_result: AnalysisResult) -> go.Figure:
        """Create ML model feature importance visualization."""
        
        feature_data = self._prepare_feature_importance_data(analysis_result)
        
        if feature_data.empty:
            return self._create_empty_chart("No ML feature data available")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=feature_data['importance'],
            y=feature_data['feature'],
            orientation='h',
            marker_color=self.color_palette['secondary'],
            text=feature_data['importance'].round(3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'ML Model Feature Importance',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            height=self.default_height,
            template=self.plotly_theme,
            showlegend=False
        )
        
        return fig
    
    def _prepare_risk_correlation_data(self, analysis_result: AnalysisResult) -> pd.DataFrame:
        """Prepare data for risk correlation heatmap."""
        
        # Create synthetic correlation data based on suppliers and risk events
        data = []
        
        for supplier in analysis_result.suppliers:
            supplier_risks = [re for re in analysis_result.risk_events if re.supplier_id == supplier.id]
            
            row = {
                'financial_health': supplier.financial_health_score or 50,
                'criticality': supplier.criticality_score,
                'geographic_risk': self._get_geographic_risk_score(supplier.country),
                'industry_risk': self._get_industry_risk_score(supplier.industry),
                'risk_events': len(supplier_risks),
                'overall_risk': sum(re.calculate_risk_score() for re in supplier_risks) / max(len(supplier_risks), 1) * 10
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _prepare_geographic_data(self, analysis_result: AnalysisResult) -> pd.DataFrame:
        """Prepare geographic data for map visualization."""
        
        # Group suppliers by country
        country_data = {}
        
        for supplier in analysis_result.suppliers:
            country = supplier.country
            if country not in country_data:
                country_data[country] = {
                    'supplier_count': 0,
                    'risk_scores': [],
                    'high_risk_count': 0
                }
            
            country_data[country]['supplier_count'] += 1
            
            # Calculate risk score for supplier
            supplier_risks = [re for re in analysis_result.risk_events if re.supplier_id == supplier.id]
            if supplier_risks:
                risk_score = max(re.calculate_risk_score() * 10 for re in supplier_risks)
                country_data[country]['risk_scores'].append(risk_score)
                if risk_score > 70:
                    country_data[country]['high_risk_count'] += 1
            else:
                country_data[country]['risk_scores'].append(supplier.criticality_score * 0.5)
        
        # Convert to DataFrame
        geo_data = []
        for country, data in country_data.items():
            geo_data.append({
                'country': country,
                'country_code': self._get_country_code(country),
                'supplier_count': data['supplier_count'],
                'avg_risk_score': np.mean(data['risk_scores']) if data['risk_scores'] else 0,
                'high_risk_count': data['high_risk_count']
            })
        
        return pd.DataFrame(geo_data)
    
    def _generate_risk_trend_data(self, analysis_result: AnalysisResult, 
                                 time_period: str) -> pd.DataFrame:
        """Generate time series data for risk trends."""
        
        # Parse time period
        days = int(time_period.replace('d', ''))
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic trend data
        base_risk = analysis_result.overall_risk_score
        high_risk_count = len(analysis_result.high_risk_suppliers)
        risk_event_count = len(analysis_result.risk_events)
        
        trend_data = []
        for i, date in enumerate(dates):
            # Add some realistic variation
            risk_variation = np.sin(i * 0.1) * 5 + np.random.normal(0, 2)
            count_variation = max(0, int(np.random.normal(0, 1)))
            
            trend_data.append({
                'date': date,
                'overall_risk': max(0, min(100, base_risk + risk_variation)),
                'high_risk_suppliers': max(0, high_risk_count + count_variation),
                'risk_events': max(0, risk_event_count + count_variation)
            })
        
        return pd.DataFrame(trend_data)
    
    def _prepare_supplier_scatter_data(self, analysis_result: AnalysisResult) -> pd.DataFrame:
        """Prepare data for supplier risk scatter plot."""
        
        scatter_data = []
        
        for supplier in analysis_result.suppliers:
            supplier_risks = [re for re in analysis_result.risk_events if re.supplier_id == supplier.id]
            
            # Calculate risk score and level
            if supplier_risks:
                risk_score = max(re.calculate_risk_score() * 10 for re in supplier_risks)
                risk_level = max(re.severity for re in supplier_risks)
            else:
                risk_score = supplier.criticality_score * 0.5
                risk_level = 'low' if risk_score < 40 else 'medium'
            
            # Size based on annual revenue
            size = 20
            if supplier.annual_revenue:
                size = min(40, max(10, supplier.annual_revenue / 10000000))
            
            scatter_data.append({
                'name': supplier.name,
                'country': supplier.country,
                'industry': supplier.industry,
                'criticality_score': supplier.criticality_score,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'size': size
            })
        
        return pd.DataFrame(scatter_data)
    
    def _prepare_industry_data(self, analysis_result: AnalysisResult) -> pd.DataFrame:
        """Prepare industry risk data."""
        
        industry_data = {}
        
        for supplier in analysis_result.suppliers:
            industry = supplier.industry
            if industry not in industry_data:
                industry_data[industry] = {
                    'supplier_count': 0,
                    'risk_scores': [],
                    'high_risk_count': 0
                }
            
            industry_data[industry]['supplier_count'] += 1
            
            # Calculate risk score
            supplier_risks = [re for re in analysis_result.risk_events if re.supplier_id == supplier.id]
            if supplier_risks:
                risk_score = max(re.calculate_risk_score() * 10 for re in supplier_risks)
                industry_data[industry]['risk_scores'].append(risk_score)
                if risk_score > 70:
                    industry_data[industry]['high_risk_count'] += 1
            else:
                industry_data[industry]['risk_scores'].append(supplier.criticality_score * 0.5)
        
        # Convert to DataFrame
        result_data = []
        for industry, data in industry_data.items():
            result_data.append({
                'industry': industry,
                'supplier_count': data['supplier_count'],
                'avg_risk_score': np.mean(data['risk_scores']) if data['risk_scores'] else 0,
                'high_risk_count': data['high_risk_count']
            })
        
        return pd.DataFrame(result_data).sort_values('avg_risk_score', ascending=False)
    
    def _prepare_feature_importance_data(self, analysis_result: AnalysisResult) -> pd.DataFrame:
        """Prepare ML feature importance data."""
        
        # Aggregate feature importance from ML predictions
        feature_importance = {}
        
        for prediction in analysis_result.ml_predictions:
            for feature, importance in prediction.feature_importance.items():
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].append(importance)
        
        # Calculate average importance
        avg_importance = {}
        for feature, importances in feature_importance.items():
            avg_importance[feature] = np.mean(importances)
        
        # Convert to DataFrame and sort
        feature_data = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in avg_importance.items()
        ]).sort_values('importance', ascending=True)
        
        return feature_data
    
    def _get_geographic_risk_score(self, country: str) -> float:
        """Get geographic risk score for a country."""
        # Simplified risk scoring based on country
        risk_scores = {
            'Germany': 20, 'Netherlands': 15, 'United States': 25,
            'Taiwan': 45, 'Brazil': 55, 'China': 50, 'India': 60
        }
        return risk_scores.get(country, 40)
    
    def _get_industry_risk_score(self, industry: str) -> float:
        """Get industry risk score."""
        risk_scores = {
            'Manufacturing': 50, 'Electronics': 60, 'Logistics': 40,
            'Materials': 55, 'Mining': 70, 'Technology': 35
        }
        return risk_scores.get(industry, 45)
    
    def _get_country_code(self, country: str) -> str:
        """Get ISO country code for mapping."""
        country_codes = {
            'Germany': 'DEU', 'Netherlands': 'NLD', 'United States': 'USA',
            'Taiwan': 'TWN', 'Brazil': 'BRA', 'China': 'CHN', 'India': 'IND'
        }
        return country_codes.get(country, 'USA')
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=self.default_height,
            template=self.plotly_theme,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig