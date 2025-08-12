"""
Risk event data models with comprehensive validation.
"""

from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime


class RiskLevel(str, Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataSource(str, Enum):
    """Data source types for risk events."""
    WEB_SCRAPING = "web_scraping"
    API_DATA = "api_data"
    FILE_UPLOAD = "file_upload"
    MANUAL_INPUT = "manual_input"
    ML_PREDICTION = "ml_prediction"
    AGENT_ANALYSIS = "agent_analysis"


class RiskEvent(BaseModel):
    """
    Risk event model representing potential supply chain disruptions.
    """
    
    event_id: str = Field(..., description="Unique risk event identifier")
    supplier_id: str = Field(..., description="Associated supplier ID")
    event_type: Literal["geopolitical", "financial", "environmental", "operational", "regulatory", "cyber"] = Field(
        ..., description="Type of risk event"
    )
    severity: RiskLevel = Field(..., description="Risk severity level")
    probability: float = Field(..., ge=0, le=1, description="ML predicted probability (0-1)")
    impact_score: float = Field(..., ge=0, le=10, description="Business impact score (0-10)")
    confidence_level: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    description: str = Field(..., min_length=10, description="Detailed risk event description")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    data_sources: List[DataSource] = Field(default_factory=list, description="Data sources used")
    detected_at: datetime = Field(default_factory=datetime.now, description="Detection timestamp")
    geographic_scope: Optional[str] = Field(None, description="Geographic scope of impact")
    predicted_timeline: Optional[str] = Field(None, description="Predicted timeline for risk materialization")
    mitigation_actions: List[str] = Field(default_factory=list, description="Recommended mitigation actions")
    status: Literal["active", "monitoring", "resolved", "false_positive"] = Field(
        default="active", description="Current status of risk event"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @validator('event_id')
    def validate_event_id(cls, v):
        """Ensure event ID follows proper format."""
        if not v.startswith('RISK_'):
            v = f'RISK_{v}'
        return v.upper()
    
    @validator('description')
    def validate_description(cls, v):
        """Ensure description is meaningful."""
        if len(v.strip()) < 10:
            raise ValueError('Description must be at least 10 characters long')
        return v.strip()
    
    @validator('evidence')
    def validate_evidence(cls, v):
        """Clean and validate evidence entries."""
        return [item.strip() for item in v if item.strip()]
    
    @validator('mitigation_actions')
    def validate_mitigation_actions(cls, v):
        """Clean and validate mitigation actions."""
        return [action.strip() for action in v if action.strip()]
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk score based on probability, impact, and confidence."""
        return (self.probability * self.impact_score * self.confidence_level) / 10.0
    
    def is_high_priority(self) -> bool:
        """Determine if this is a high priority risk event."""
        return (
            self.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            self.calculate_risk_score() > 0.7 or
            self.impact_score > 8.0
        )
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "event_id": "RISK_001",
                "supplier_id": "SUP001",
                "event_type": "geopolitical",
                "severity": "high",
                "probability": 0.75,
                "impact_score": 8.5,
                "confidence_level": 0.85,
                "description": "Political instability in supplier's region may disrupt operations",
                "evidence": ["News reports of regional tensions", "Government policy changes"],
                "data_sources": ["web_scraping", "api_data"],
                "geographic_scope": "Eastern Europe",
                "predicted_timeline": "2-4 weeks",
                "mitigation_actions": ["Identify alternative suppliers", "Increase inventory buffer"]
            }
        }