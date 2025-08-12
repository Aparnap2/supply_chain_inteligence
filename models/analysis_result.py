"""
Comprehensive analysis result model for complete supply chain assessment.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from .supplier import Supplier
from .risk_event import RiskEvent
from .scraped_data import ScrapedData
from .ml_prediction import MLPrediction


class AnalysisResult(BaseModel):
    """
    Comprehensive analysis result containing all supply chain intelligence data.
    """
    
    analysis_id: str = Field(..., description="Unique analysis identifier")
    suppliers: List[Supplier] = Field(..., description="Analyzed suppliers")
    scraped_data: List[ScrapedData] = Field(default_factory=list, description="Web-scraped data")
    risk_events: List[RiskEvent] = Field(default_factory=list, description="Identified risk events")
    ml_predictions: List[MLPrediction] = Field(default_factory=list, description="ML model predictions")
    overall_risk_score: float = Field(..., ge=0, le=100, description="Overall portfolio risk score (0-100)")
    high_risk_suppliers: List[str] = Field(default_factory=list, description="High-risk supplier IDs")
    critical_risk_events: List[str] = Field(default_factory=list, description="Critical risk event IDs")
    recommendations: List[str] = Field(default_factory=list, description="Strategic recommendations")
    data_quality_score: float = Field(default=0.8, ge=0, le=1, description="Overall data quality (0-1)")
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")
    workflow_status: str = Field(default="completed", description="Workflow execution status")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    generated_at: datetime = Field(default_factory=datetime.now, description="Analysis generation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Analysis expiration timestamp")
    
    @validator('analysis_id')
    def validate_analysis_id(cls, v):
        """Ensure analysis ID follows proper format."""
        if not v.startswith('ANALYSIS_'):
            v = f'ANALYSIS_{v}'
        return v.upper()
    
    @validator('suppliers')
    def validate_suppliers(cls, v):
        """Ensure at least one supplier is provided."""
        if not v:
            raise ValueError('At least one supplier must be provided')
        return v
    
    @validator('recommendations')
    def validate_recommendations(cls, v):
        """Clean and validate recommendations."""
        return [rec.strip() for rec in v if rec.strip()]
    
    @validator('high_risk_suppliers')
    def validate_high_risk_suppliers(cls, v):
        """Ensure high-risk supplier IDs are valid."""
        return [supplier_id.strip() for supplier_id in v if supplier_id.strip()]
    
    @validator('critical_risk_events')
    def validate_critical_risk_events(cls, v):
        """Ensure critical risk event IDs are valid."""
        return [event_id.strip() for event_id in v if event_id.strip()]
    
    def get_supplier_count(self) -> int:
        """Get total number of suppliers analyzed."""
        return len(self.suppliers)
    
    def get_risk_event_count(self) -> int:
        """Get total number of risk events identified."""
        return len(self.risk_events)
    
    def get_high_risk_count(self) -> int:
        """Get number of high-risk suppliers."""
        return len(self.high_risk_suppliers)
    
    def get_critical_event_count(self) -> int:
        """Get number of critical risk events."""
        return len(self.critical_risk_events)
    
    def get_data_coverage_score(self) -> float:
        """Calculate data coverage score based on available data."""
        total_possible_data_points = len(self.suppliers) * 3  # scraped, ml, risk events
        actual_data_points = len(self.scraped_data) + len(self.ml_predictions) + len(self.risk_events)
        
        if total_possible_data_points == 0:
            return 0.0
        
        return min(actual_data_points / total_possible_data_points, 1.0)
    
    def get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of risk levels across suppliers."""
        distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for risk_event in self.risk_events:
            if risk_event.severity.value in distribution:
                distribution[risk_event.severity.value] += 1
        
        return distribution
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics for the analysis."""
        return {
            "total_suppliers": self.get_supplier_count(),
            "total_risk_events": self.get_risk_event_count(),
            "high_risk_suppliers": self.get_high_risk_count(),
            "critical_events": self.get_critical_event_count(),
            "overall_risk_score": self.overall_risk_score,
            "data_quality_score": self.data_quality_score,
            "data_coverage_score": self.get_data_coverage_score(),
            "risk_distribution": self.get_risk_distribution(),
            "processing_time": self.processing_time,
            "generated_at": self.generated_at
        }
    
    def is_analysis_current(self, max_age_hours: int = 24) -> bool:
        """Check if analysis is still current based on age."""
        if self.expires_at:
            return datetime.now() < self.expires_at
        
        age_hours = (datetime.now() - self.generated_at).total_seconds() / 3600
        return age_hours <= max_age_hours
    
    def get_top_recommendations(self, n: int = 5) -> List[str]:
        """Get top N recommendations."""
        return self.recommendations[:n]
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "analysis_id": "ANALYSIS_001",
                "suppliers": [],  # Would contain Supplier objects
                "overall_risk_score": 75.5,
                "high_risk_suppliers": ["SUP001", "SUP003"],
                "critical_risk_events": ["RISK_001"],
                "recommendations": [
                    "Diversify supplier base in high-risk regions",
                    "Increase inventory buffers for critical components",
                    "Implement enhanced monitoring for tier-1 suppliers"
                ],
                "data_quality_score": 0.85,
                "workflow_status": "completed",
                "processing_time": 45.2
            }
        }