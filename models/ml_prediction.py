"""
Machine learning prediction model with confidence scoring.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from .risk_event import RiskLevel


class MLPrediction(BaseModel):
    """
    Machine learning prediction results with confidence and feature importance.
    """
    
    prediction_id: str = Field(..., description="Unique prediction identifier")
    supplier_id: str = Field(..., description="Associated supplier ID")
    model_version: str = Field(..., description="ML model version used")
    prediction_type: str = Field(..., description="Type of prediction (risk_level, impact_score, probability)")
    prediction: Any = Field(..., description="Prediction result")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score (0-1)")
    probability_scores: Dict[str, float] = Field(default_factory=dict, description="Class probability scores")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    input_features: Dict[str, Any] = Field(default_factory=dict, description="Input features used")
    model_metadata: Dict[str, Any] = Field(default_factory=dict, description="Model metadata")
    predicted_at: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    is_valid: bool = Field(default=True, description="Prediction validity flag")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    
    @validator('prediction_id')
    def validate_prediction_id(cls, v):
        """Ensure prediction ID follows proper format."""
        if not v.startswith('PRED_'):
            v = f'PRED_{v}'
        return v.upper()
    
    @validator('probability_scores')
    def validate_probability_scores(cls, v):
        """Validate probability scores sum to 1 for classification."""
        if v and abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError('Probability scores must sum to approximately 1.0')
        return v
    
    @validator('feature_importance')
    def validate_feature_importance(cls, v):
        """Ensure feature importance scores are valid."""
        for feature, importance in v.items():
            if not isinstance(importance, (int, float)):
                raise ValueError(f'Feature importance for {feature} must be numeric')
        return v
    
    def get_risk_level(self) -> Optional[RiskLevel]:
        """Convert prediction to risk level if applicable."""
        if self.prediction_type == "risk_level":
            if isinstance(self.prediction, str):
                try:
                    return RiskLevel(self.prediction.lower())
                except ValueError:
                    return None
        return None
    
    def get_top_features(self, n: int = 5) -> List[tuple]:
        """Get top N most important features."""
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if prediction has high confidence."""
        return self.confidence >= threshold
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get a summary of the prediction."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "risk_level": self.get_risk_level(),
            "top_features": self.get_top_features(3),
            "is_high_confidence": self.is_high_confidence(),
            "predicted_at": self.predicted_at
        }
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "prediction_id": "PRED_001",
                "supplier_id": "SUP001",
                "model_version": "v1.2.0",
                "prediction_type": "risk_level",
                "prediction": "high",
                "confidence": 0.85,
                "probability_scores": {
                    "low": 0.05,
                    "medium": 0.10,
                    "high": 0.75,
                    "critical": 0.10
                },
                "feature_importance": {
                    "financial_health_score": 0.35,
                    "geographic_risk": 0.25,
                    "industry_volatility": 0.20,
                    "sentiment_score": 0.20
                },
                "input_features": {
                    "financial_health_score": 65.0,
                    "geographic_risk": 0.7,
                    "industry_volatility": 0.6
                }
            }
        }