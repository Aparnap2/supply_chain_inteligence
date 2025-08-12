"""
Utility functions for the Streamlit dashboard.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import streamlit as st

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.analysis_result import AnalysisResult
from models.supplier import Supplier
from models.risk_event import RiskEvent, RiskLevel
from models.scraped_data import ScrapedData
from models.ml_prediction import MLPrediction


class DataLoader:
    """Utility class for loading and managing dashboard data."""
    
    @staticmethod
    def load_analysis_results(file_path: str) -> Optional[AnalysisResult]:
        """Load analysis results from JSON file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return AnalysisResult(**data)
        except Exception as e:
            st.error(f"Error loading analysis results: {str(e)}")
        return None
    
    @staticmethod
    def save_analysis_results(analysis_result: AnalysisResult, file_path: str) -> bool:
        """Save analysis results to JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(analysis_result.dict(), f, indent=2, default=str)
            return True
        except Exception as e:
            st.error(f"Error saving analysis results: {str(e)}")
            return False
    
    @staticmethod
    def load_suppliers_from_csv(file_path: str) -> List[Supplier]:
        """Load suppliers from CSV file."""
        try:
            df = pd.read_csv(file_path)
            suppliers = []
            
            for _, row in df.iterrows():
                supplier_data = {
                    'id': row.get('id', f'SUP_{len(suppliers)+1:03d}'),
                    'name': row['name'],
                    'country': row.get('country', 'Unknown'),
                    'region': row.get('region', 'Unknown'),
                    'industry': row.get('industry', 'Unknown'),
                    'website': row.get('website'),
                    'annual_revenue': row.get('annual_revenue'),
                    'employee_count': row.get('employee_count'),
                    'financial_health_score': row.get('financial_health_score'),
                    'tier': row.get('tier', 'tier_2'),
                    'criticality_score': row.get('criticality_score', 50.0)
                }
                
                # Remove None values
                supplier_data = {k: v for k, v in supplier_data.items() if pd.notna(v)}
                suppliers.append(Supplier(**supplier_data))
            
            return suppliers
        except Exception as e:
            st.error(f"Error loading suppliers from CSV: {str(e)}")
            return []
    
    @staticmethod
    def create_demo_analysis_result() -> AnalysisResult:
        """Create a demo analysis result for testing."""
        # Create demo suppliers
        suppliers = [
            Supplier(
                id="SUP001",
                name="Global Manufacturing Corp",
                website="https://globalmanufacturing.com",
                country="Germany",
                region="Europe",
                industry="Manufacturing",
                annual_revenue=500000000.0,
                employee_count=2500,
                financial_health_score=85.0,
                tier="tier_1",
                criticality_score=90.0
            ),
            Supplier(
                id="SUP002",
                name="Asia Pacific Electronics",
                website="https://apelectronics.com",
                country="Taiwan",
                region="Asia Pacific",
                industry="Electronics",
                annual_revenue=300000000.0,
                employee_count=1800,
                financial_health_score=72.0,
                tier="tier_1",
                criticality_score=85.0
            ),
            Supplier(
                id="SUP003",
                name="European Logistics Solutions",
                website="https://eurolog.com",
                country="Netherlands",
                region="Europe",
                industry="Logistics",
                annual_revenue=150000000.0,
                employee_count=800,
                financial_health_score=88.0,
                tier="tier_2",
                criticality_score=60.0
            )
        ]
        
        # Create demo risk events
        risk_events = [
            RiskEvent(
                event_id="RISK_001",
                supplier_id="SUP001",
                event_type="geopolitical",
                severity=RiskLevel.HIGH,
                probability=0.75,
                impact_score=8.5,
                confidence_level=0.85,
                description="Political instability in supplier's region may disrupt operations",
                evidence=["News reports of regional tensions", "Government policy changes"],
                geographic_scope="Eastern Europe",
                predicted_timeline="2-4 weeks",
                mitigation_actions=["Identify alternative suppliers", "Increase inventory buffer"]
            ),
            RiskEvent(
                event_id="RISK_002",
                supplier_id="SUP002",
                event_type="financial",
                severity=RiskLevel.MEDIUM,
                probability=0.45,
                impact_score=6.2,
                confidence_level=0.78,
                description="Supplier showing signs of financial stress based on recent reports",
                evidence=["Delayed payments to vendors", "Credit rating downgrade"],
                geographic_scope="Asia Pacific",
                predicted_timeline="1-2 months",
                mitigation_actions=["Monitor financial health", "Prepare backup suppliers"]
            )
        ]
        
        # Create demo ML predictions
        ml_predictions = [
            MLPrediction(
                prediction_id="PRED_001",
                supplier_id="SUP001",
                model_version="v1.2.0",
                prediction_type="risk_level",
                prediction="high",
                confidence=0.82,
                feature_importance={"financial_health": 0.35, "geographic_risk": 0.28, "industry_risk": 0.37},
                model_metadata={"model_type": "RandomForest", "training_date": "2024-01-15"}
            ),
            MLPrediction(
                prediction_id="PRED_002",
                supplier_id="SUP002",
                model_version="v1.2.0",
                prediction_type="risk_level",
                prediction="medium",
                confidence=0.76,
                feature_importance={"financial_health": 0.42, "geographic_risk": 0.31, "industry_risk": 0.27},
                model_metadata={"model_type": "RandomForest", "training_date": "2024-01-15"}
            )
        ]
        
        # Create analysis result
        return AnalysisResult(
            analysis_id="ANALYSIS_DEMO_001",
            suppliers=suppliers,
            risk_events=risk_events,
            ml_predictions=ml_predictions,
            overall_risk_score=68.5,
            high_risk_suppliers=["SUP001"],
            critical_risk_events=["RISK_001"],
            recommendations=[
                "Diversify supplier base in high-risk regions",
                "Increase inventory buffers for critical components",
                "Implement enhanced monitoring for tier-1 suppliers",
                "Develop contingency plans for geopolitical risks"
            ],
            data_quality_score=0.87,
            workflow_status="completed",
            processing_time=45.2
        )


class DashboardHelpers:
    """Helper functions for dashboard components."""
    
    @staticmethod
    def format_risk_level(risk_level: str) -> str:
        """Format risk level with appropriate styling."""
        risk_colors = {
            "low": "🟢",
            "medium": "🟡", 
            "high": "🟠",
            "critical": "🔴"
        }
        return f"{risk_colors.get(risk_level.lower(), '⚪')} {risk_level.title()}"
    
    @staticmethod
    def format_currency(amount: Optional[float]) -> str:
        """Format currency amounts."""
        if amount is None:
            return "N/A"
        
        if amount >= 1_000_000_000:
            return f"${amount/1_000_000_000:.1f}B"
        elif amount >= 1_000_000:
            return f"${amount/1_000_000:.1f}M"
        elif amount >= 1_000:
            return f"${amount/1_000:.1f}K"
        else:
            return f"${amount:.0f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format percentage values."""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def get_risk_color(risk_score: float) -> str:
        """Get color based on risk score."""
        if risk_score >= 80:
            return "#d62728"  # Red
        elif risk_score >= 60:
            return "#ff7f0e"  # Orange
        elif risk_score >= 40:
            return "#ffbb78"  # Light orange
        else:
            return "#2ca02c"  # Green
    
    @staticmethod
    def calculate_trend(current: float, previous: float) -> Dict[str, Any]:
        """Calculate trend information."""
        if previous == 0:
            return {"direction": "neutral", "percentage": 0, "arrow": "→"}
        
        change = ((current - previous) / previous) * 100
        
        if change > 0:
            return {"direction": "up", "percentage": change, "arrow": "↗"}
        elif change < 0:
            return {"direction": "down", "percentage": abs(change), "arrow": "↘"}
        else:
            return {"direction": "neutral", "percentage": 0, "arrow": "→"}
    
    @staticmethod
    def create_summary_card(title: str, value: str, delta: Optional[str] = None, 
                          help_text: Optional[str] = None) -> str:
        """Create HTML for a summary card."""
        delta_html = f'<div class="delta">{delta}</div>' if delta else ''
        help_html = f'<div class="help-text">{help_text}</div>' if help_text else ''
        
        return f"""
        <div class="summary-card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            {delta_html}
            {help_html}
        </div>
        """
    
    @staticmethod
    def filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to a DataFrame."""
        filtered_df = df.copy()
        
        for column, filter_value in filters.items():
            if filter_value and filter_value != "All":
                if column in df.columns:
                    if isinstance(filter_value, str):
                        filtered_df = filtered_df[
                            filtered_df[column].str.contains(filter_value, case=False, na=False)
                        ]
                    else:
                        filtered_df = filtered_df[filtered_df[column] == filter_value]
        
        return filtered_df
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
        """Export DataFrame to CSV bytes."""
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def export_to_json(data: Dict[str, Any], filename: str) -> bytes:
        """Export data to JSON bytes."""
        return json.dumps(data, indent=2, default=str).encode('utf-8')


class CacheManager:
    """Manage caching for dashboard data."""
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_cached_analysis_results(file_path: str) -> Optional[AnalysisResult]:
        """Load analysis results with caching."""
        return DataLoader.load_analysis_results(file_path)
    
    @staticmethod
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def load_cached_suppliers(file_path: str) -> List[Supplier]:
        """Load suppliers with caching."""
        return DataLoader.load_suppliers_from_csv(file_path)
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_cached_demo_data() -> AnalysisResult:
        """Get demo data with caching."""
        return DataLoader.create_demo_analysis_result()
    
    @staticmethod
    def clear_cache():
        """Clear all cached data."""
        st.cache_data.clear()


# Configuration constants
DASHBOARD_CONFIG = {
    "refresh_intervals": [30, 60, 300, 600, 1800],  # seconds
    "risk_thresholds": {
        "low": 30,
        "medium": 60,
        "high": 80,
        "critical": 90
    },
    "chart_colors": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "success": "#2ca02c",
        "warning": "#ff7f0e",
        "danger": "#d62728"
    },
    "default_page_size": 20,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "supported_file_types": ["csv", "xlsx", "json"]
}