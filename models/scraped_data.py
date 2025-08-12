"""
Scraped data model for web-collected information.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from .risk_event import DataSource


class ScrapedData(BaseModel):
    """
    Model for web-scraped content with metadata and quality scoring.
    """
    
    data_id: str = Field(..., description="Unique scraped data identifier")
    supplier_id: str = Field(..., description="Associated supplier ID")
    source_url: str = Field(..., description="Source URL of scraped content")
    content: str = Field(..., min_length=1, description="Scraped content text")
    content_type: str = Field(default="text/html", description="Content type")
    extraction_method: str = Field(..., description="Extraction method used")
    relevance_score: float = Field(default=0.5, ge=0, le=1, description="Content relevance score (0-1)")
    quality_score: float = Field(default=0.5, ge=0, le=1, description="Data quality score (0-1)")
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Sentiment analysis score (-1 to 1)")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    entities: List[str] = Field(default_factory=list, description="Named entities found")
    risk_indicators: List[str] = Field(default_factory=list, description="Risk-related indicators")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    scraped_at: datetime = Field(default_factory=datetime.now, description="Scraping timestamp")
    data_source: DataSource = Field(default=DataSource.WEB_SCRAPING, description="Data source type")
    processing_status: str = Field(default="raw", description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if scraping failed")
    
    @validator('data_id')
    def validate_data_id(cls, v):
        """Ensure data ID follows proper format."""
        if not v.startswith('DATA_'):
            v = f'DATA_{v}'
        return v.upper()
    
    @validator('source_url')
    def validate_source_url(cls, v):
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        """Ensure content is not empty."""
        content = v.strip()
        if not content:
            raise ValueError('Content cannot be empty')
        return content
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Clean and validate keywords."""
        return [keyword.strip().lower() for keyword in v if keyword.strip()]
    
    @validator('entities')
    def validate_entities(cls, v):
        """Clean and validate entities."""
        return [entity.strip() for entity in v if entity.strip()]
    
    @validator('risk_indicators')
    def validate_risk_indicators(cls, v):
        """Clean and validate risk indicators."""
        return [indicator.strip().lower() for indicator in v if indicator.strip()]
    
    def calculate_overall_score(self) -> float:
        """Calculate overall data quality score."""
        return (self.relevance_score + self.quality_score) / 2.0
    
    def has_risk_signals(self) -> bool:
        """Check if data contains risk signals."""
        return len(self.risk_indicators) > 0 or (
            self.sentiment_score is not None and self.sentiment_score < -0.3
        )
    
    def get_content_summary(self, max_length: int = 200) -> str:
        """Get a summary of the content."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "data_id": "DATA_001",
                "supplier_id": "SUP001",
                "source_url": "https://example-supplier.com/news",
                "content": "Company announces expansion plans despite regional challenges...",
                "extraction_method": "LLMExtractionStrategy",
                "relevance_score": 0.85,
                "quality_score": 0.90,
                "sentiment_score": 0.2,
                "keywords": ["expansion", "growth", "challenges"],
                "entities": ["Company Name", "Region"],
                "risk_indicators": ["supply chain disruption", "regulatory changes"],
                "metadata": {"page_title": "Company News", "word_count": 450}
            }
        }