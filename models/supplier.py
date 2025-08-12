"""
Supplier data model with comprehensive validation.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime


class Supplier(BaseModel):
    """
    Core supplier information model with validation.
    Represents a supply chain partner with risk assessment data.
    """
    
    id: str = Field(..., description="Unique supplier identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Company name")
    website: Optional[str] = Field(None, description="Company website for scraping")
    country: str = Field(..., description="Primary operating country")
    region: str = Field(..., description="Geographic region")
    industry: str = Field(..., description="Primary industry sector")
    annual_revenue: Optional[float] = Field(None, ge=0, description="Annual revenue in USD")
    employee_count: Optional[int] = Field(None, ge=1, description="Number of employees")
    financial_health_score: Optional[float] = Field(None, ge=0, le=100, description="Financial health score (0-100)")
    tier: Literal["tier_1", "tier_2", "tier_3"] = Field(default="tier_2", description="Supplier tier classification")
    criticality_score: float = Field(default=50.0, ge=0, le=100, description="Business criticality score (0-100)")
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @validator('website')
    def validate_website(cls, v):
        """Validate website URL format."""
        if v is not None and v.strip():
            if not v.startswith(('http://', 'https://')):
                v = f'https://{v}'
            # Basic URL validation
            if not any(char in v for char in ['.', '/']):
                raise ValueError('Invalid website URL format')
        return v
    
    @validator('country')
    def validate_country(cls, v):
        """Ensure country name is properly formatted."""
        return v.strip().title()
    
    @validator('region')
    def validate_region(cls, v):
        """Ensure region name is properly formatted."""
        return v.strip().title()
    
    @validator('industry')
    def validate_industry(cls, v):
        """Ensure industry name is properly formatted."""
        return v.strip().title()
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "SUP001",
                "name": "Global Manufacturing Corp",
                "website": "https://globalmanufacturing.com",
                "country": "Germany",
                "region": "Europe",
                "industry": "Manufacturing",
                "annual_revenue": 500000000.0,
                "employee_count": 2500,
                "financial_health_score": 85.0,
                "tier": "tier_1",
                "criticality_score": 90.0
            }
        }