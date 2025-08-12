"""
File upload and manual data input handlers with validation and quality checks.
"""

import pandas as pd
import numpy as np
import json
import csv
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
import uuid
from io import StringIO, BytesIO

from pydantic import BaseModel, Field, ValidationError, validator
from models.supplier import Supplier
from models.scraped_data import ScrapedData


logger = logging.getLogger(__name__)


class DataLineage(BaseModel):
    """Data lineage tracking for audit trails."""
    data_id: str = Field(..., description="Unique data identifier")
    source_file: Optional[str] = Field(None, description="Source file name")
    upload_timestamp: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    user_id: Optional[str] = Field(None, description="User who uploaded the data")
    processing_steps: List[str] = Field(default_factory=list, description="Processing steps applied")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")
    quality_score: float = Field(default=0.0, ge=0, le=1, description="Overall quality score")
    row_count: int = Field(default=0, description="Number of data rows")
    column_count: int = Field(default=0, description="Number of columns")
    file_size_bytes: Optional[int] = Field(None, description="Original file size in bytes")


class SupplierDataInput(BaseModel):
    """Manual supplier data input with validation."""
    name: str = Field(..., min_length=1, max_length=200, description="Company name")
    website: Optional[str] = Field(None, description="Company website")
    country: str = Field(..., description="Primary operating country")
    region: str = Field(..., description="Geographic region")
    industry: str = Field(..., description="Primary industry sector")
    annual_revenue: Optional[float] = Field(None, ge=0, description="Annual revenue in USD")
    employee_count: Optional[int] = Field(None, ge=1, description="Number of employees")
    financial_health_score: Optional[float] = Field(None, ge=0, le=100, description="Financial health score")
    tier: str = Field(default="tier_2", description="Supplier tier classification")
    criticality_score: float = Field(default=50.0, ge=0, le=100, description="Business criticality score")
    contact_email: Optional[str] = Field(None, description="Contact email")
    contact_phone: Optional[str] = Field(None, description="Contact phone")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    @validator('website')
    def validate_website(cls, v):
        """Validate website URL format."""
        if v is not None and v.strip():
            if not v.startswith(('http://', 'https://')):
                v = f'https://{v}'
        return v
    
    @validator('tier')
    def validate_tier(cls, v):
        """Validate tier classification."""
        valid_tiers = ["tier_1", "tier_2", "tier_3"]
        if v not in valid_tiers:
            raise ValueError(f"Tier must be one of {valid_tiers}")
        return v
    
    def to_supplier(self, supplier_id: Optional[str] = None) -> Supplier:
        """Convert to Supplier model."""
        if not supplier_id:
            supplier_id = f"SUP{uuid.uuid4().hex[:6].upper()}"
        
        return Supplier(
            id=supplier_id,
            name=self.name,
            website=self.website,
            country=self.country,
            region=self.region,
            industry=self.industry,
            annual_revenue=self.annual_revenue,
            employee_count=self.employee_count,
            financial_health_score=self.financial_health_score,
            tier=self.tier,
            criticality_score=self.criticality_score
        )


class RiskEventInput(BaseModel):
    """Manual risk event data input with validation."""
    supplier_id: str = Field(..., description="Associated supplier ID")
    event_type: str = Field(..., description="Type of risk event")
    severity: str = Field(..., description="Event severity level")
    description: str = Field(..., min_length=10, description="Event description")
    location: Optional[str] = Field(None, description="Event location")
    impact_assessment: Optional[str] = Field(None, description="Impact assessment")
    mitigation_actions: Optional[str] = Field(None, description="Mitigation actions taken")
    event_date: Optional[str] = Field(None, description="Event date (YYYY-MM-DD)")
    resolution_date: Optional[str] = Field(None, description="Resolution date (YYYY-MM-DD)")
    financial_impact: Optional[float] = Field(None, description="Financial impact in USD")
    
    @validator('severity')
    def validate_severity(cls, v):
        """Validate severity level."""
        valid_severities = ["low", "medium", "high", "critical"]
        if v.lower() not in valid_severities:
            raise ValueError(f"Severity must be one of {valid_severities}")
        return v.lower()
    
    @validator('event_date', 'resolution_date')
    def validate_dates(cls, v):
        """Validate date format."""
        if v is not None and v.strip():
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class FileInputHandler:
    """
    Handler for file uploads and manual data input with comprehensive validation.
    """
    
    def __init__(self, max_file_size_mb: int = 50, allowed_extensions: List[str] = None):
        """
        Initialize the file input handler.
        
        Args:
            max_file_size_mb: Maximum file size in MB
            allowed_extensions: List of allowed file extensions
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.allowed_extensions = allowed_extensions or ['.csv', '.xlsx', '.xls', '.json']
        
        # Standard column mappings for supplier data
        self.supplier_column_mappings = {
            'name': ['name', 'company_name', 'supplier_name', 'company'],
            'website': ['website', 'url', 'web_site', 'homepage'],
            'country': ['country', 'nation', 'location_country'],
            'region': ['region', 'area', 'geographic_region'],
            'industry': ['industry', 'sector', 'business_sector'],
            'annual_revenue': ['annual_revenue', 'revenue', 'yearly_revenue', 'sales'],
            'employee_count': ['employee_count', 'employees', 'staff_count', 'workforce'],
            'financial_health_score': ['financial_health_score', 'financial_score', 'health_score'],
            'tier': ['tier', 'supplier_tier', 'classification'],
            'criticality_score': ['criticality_score', 'criticality', 'importance_score']
        }
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate uploaded file.
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return False, "File does not exist"
            
            # Check file extension
            if path.suffix.lower() not in self.allowed_extensions:
                return False, f"File extension {path.suffix} not allowed. Allowed: {self.allowed_extensions}"
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_file_size_bytes:
                return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum ({self.max_file_size_bytes / 1024 / 1024}MB)"
            
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            except Exception as e:
                return False, f"File is not readable: {str(e)}"
            
            return True, "File validation passed"
            
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    def _detect_delimiter(self, file_path: str) -> str:
        """Detect CSV delimiter."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                return delimiter
        except:
            return ','  # Default to comma
    
    def _map_columns(self, df_columns: List[str], mapping_dict: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Map DataFrame columns to standard field names.
        
        Args:
            df_columns: List of DataFrame column names
            mapping_dict: Dictionary mapping standard fields to possible column names
            
        Returns:
            Dictionary mapping DataFrame columns to standard field names
        """
        column_mapping = {}
        df_columns_lower = [col.lower().strip() for col in df_columns]
        
        for standard_field, possible_names in mapping_dict.items():
            for possible_name in possible_names:
                if possible_name.lower() in df_columns_lower:
                    original_col = df_columns[df_columns_lower.index(possible_name.lower())]
                    column_mapping[original_col] = standard_field
                    break
        
        return column_mapping
    
    def _calculate_data_quality_score(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> float:
        """Calculate data quality score for the DataFrame."""
        if df.empty:
            return 0.0
        
        score = 0.0
        total_checks = 0
        
        # Completeness check
        for col in df.columns:
            if col in column_mapping:
                non_null_ratio = df[col].notna().sum() / len(df)
                score += non_null_ratio * 0.1
                total_checks += 1
        
        # Required fields check
        required_fields = ['name', 'country', 'industry']
        required_present = sum(1 for field in required_fields if field in column_mapping.values())
        score += (required_present / len(required_fields)) * 0.3
        
        # Data consistency check
        if 'annual_revenue' in column_mapping.values():
            revenue_col = next(col for col, field in column_mapping.items() if field == 'annual_revenue')
            if revenue_col in df.columns:
                numeric_ratio = pd.to_numeric(df[revenue_col], errors='coerce').notna().sum() / len(df)
                score += numeric_ratio * 0.1
        
        # Duplicate check
        if 'name' in column_mapping.values():
            name_col = next(col for col, field in column_mapping.items() if field == 'name')
            if name_col in df.columns:
                unique_ratio = df[name_col].nunique() / len(df)
                score += unique_ratio * 0.2
        
        # Format consistency
        if len(df) > 1:
            format_score = 0.2  # Assume good format if we got this far
            score += format_score
        
        return min(score, 1.0)
    
    def parse_csv_file(self, file_path: str, user_id: Optional[str] = None) -> Tuple[List[Supplier], DataLineage]:
        """
        Parse CSV file and extract supplier data.
        
        Args:
            file_path: Path to CSV file
            user_id: User ID for audit trail
            
        Returns:
            Tuple of (suppliers_list, data_lineage)
        """
        # Validate file first
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(f"File validation failed: {error_msg}")
        
        path = Path(file_path)
        file_size = path.stat().st_size
        
        # Detect delimiter and read CSV
        delimiter = self._detect_delimiter(file_path)
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin-1')
            except Exception as e:
                raise ValueError(f"Failed to read CSV file: {str(e)}")
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Map columns to standard fields
        column_mapping = self._map_columns(df.columns.tolist(), self.supplier_column_mappings)
        
        if not column_mapping:
            raise ValueError("No recognizable columns found in CSV file")
        
        # Rename columns
        df_mapped = df.rename(columns=column_mapping)
        
        # Calculate quality score
        quality_score = self._calculate_data_quality_score(df, column_mapping)
        
        # Convert to suppliers
        suppliers = []
        processing_steps = ["file_validation", "delimiter_detection", "column_mapping", "data_conversion"]
        validation_results = {"mapped_columns": column_mapping, "total_rows": len(df)}
        
        for idx, row in df_mapped.iterrows():
            try:
                # Create supplier data input
                supplier_data = {}
                for field in SupplierDataInput.__fields__.keys():
                    if field in row and pd.notna(row[field]):
                        supplier_data[field] = row[field]
                
                # Ensure required fields have defaults
                if 'name' not in supplier_data:
                    continue  # Skip rows without name
                
                supplier_data.setdefault('country', 'Unknown')
                supplier_data.setdefault('region', 'Unknown')
                supplier_data.setdefault('industry', 'Unknown')
                
                # Validate and create supplier
                supplier_input = SupplierDataInput(**supplier_data)
                supplier = supplier_input.to_supplier()
                suppliers.append(supplier)
                
            except ValidationError as e:
                logger.warning(f"Validation error for row {idx}: {e}")
                validation_results[f"row_{idx}_error"] = str(e)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                validation_results[f"row_{idx}_error"] = str(e)
        
        # Create data lineage
        data_lineage = DataLineage(
            data_id=f"UPLOAD_{uuid.uuid4().hex[:8].upper()}",
            source_file=path.name,
            user_id=user_id,
            processing_steps=processing_steps,
            validation_results=validation_results,
            quality_score=quality_score,
            row_count=len(df),
            column_count=len(df.columns),
            file_size_bytes=file_size
        )
        
        logger.info(f"Parsed CSV file: {len(suppliers)} suppliers from {len(df)} rows")
        return suppliers, data_lineage
    
    def parse_excel_file(self, file_path: str, sheet_name: Optional[str] = None, 
                        user_id: Optional[str] = None) -> Tuple[List[Supplier], DataLineage]:
        """
        Parse Excel file and extract supplier data.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Specific sheet name (optional)
            user_id: User ID for audit trail
            
        Returns:
            Tuple of (suppliers_list, data_lineage)
        """
        # Validate file first
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(f"File validation failed: {error_msg}")
        
        path = Path(file_path)
        file_size = path.stat().st_size
        
        try:
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {str(e)}")
        
        if df.empty:
            raise ValueError("Excel file is empty")
        
        # Map columns to standard fields
        column_mapping = self._map_columns(df.columns.tolist(), self.supplier_column_mappings)
        
        if not column_mapping:
            raise ValueError("No recognizable columns found in Excel file")
        
        # Rename columns
        df_mapped = df.rename(columns=column_mapping)
        
        # Calculate quality score
        quality_score = self._calculate_data_quality_score(df, column_mapping)
        
        # Convert to suppliers (similar to CSV processing)
        suppliers = []
        processing_steps = ["file_validation", "excel_parsing", "column_mapping", "data_conversion"]
        validation_results = {"mapped_columns": column_mapping, "total_rows": len(df)}
        
        for idx, row in df_mapped.iterrows():
            try:
                # Create supplier data input
                supplier_data = {}
                for field in SupplierDataInput.__fields__.keys():
                    if field in row and pd.notna(row[field]):
                        value = row[field]
                        # Handle Excel-specific data types
                        if isinstance(value, (np.integer, np.floating)):
                            value = float(value) if not np.isnan(value) else None
                        supplier_data[field] = value
                
                # Ensure required fields have defaults
                if 'name' not in supplier_data:
                    continue  # Skip rows without name
                
                supplier_data.setdefault('country', 'Unknown')
                supplier_data.setdefault('region', 'Unknown')
                supplier_data.setdefault('industry', 'Unknown')
                
                # Validate and create supplier
                supplier_input = SupplierDataInput(**supplier_data)
                supplier = supplier_input.to_supplier()
                suppliers.append(supplier)
                
            except ValidationError as e:
                logger.warning(f"Validation error for row {idx}: {e}")
                validation_results[f"row_{idx}_error"] = str(e)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                validation_results[f"row_{idx}_error"] = str(e)
        
        # Create data lineage
        data_lineage = DataLineage(
            data_id=f"UPLOAD_{uuid.uuid4().hex[:8].upper()}",
            source_file=path.name,
            user_id=user_id,
            processing_steps=processing_steps,
            validation_results=validation_results,
            quality_score=quality_score,
            row_count=len(df),
            column_count=len(df.columns),
            file_size_bytes=file_size
        )
        
        logger.info(f"Parsed Excel file: {len(suppliers)} suppliers from {len(df)} rows")
        return suppliers, data_lineage
    
    def parse_json_file(self, file_path: str, user_id: Optional[str] = None) -> Tuple[List[Supplier], DataLineage]:
        """
        Parse JSON file and extract supplier data.
        
        Args:
            file_path: Path to JSON file
            user_id: User ID for audit trail
            
        Returns:
            Tuple of (suppliers_list, data_lineage)
        """
        # Validate file first
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(f"File validation failed: {error_msg}")
        
        path = Path(file_path)
        file_size = path.stat().st_size
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to read JSON file: {str(e)}")
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'suppliers' in data:
                supplier_data = data['suppliers']
            elif 'data' in data:
                supplier_data = data['data']
            else:
                supplier_data = [data]  # Single supplier object
        elif isinstance(data, list):
            supplier_data = data
        else:
            raise ValueError("JSON file must contain an object or array")
        
        if not supplier_data:
            raise ValueError("No supplier data found in JSON file")
        
        # Convert to suppliers
        suppliers = []
        processing_steps = ["file_validation", "json_parsing", "data_validation", "supplier_creation"]
        validation_results = {"total_records": len(supplier_data)}
        
        for idx, item in enumerate(supplier_data):
            try:
                # Validate and create supplier
                supplier_input = SupplierDataInput(**item)
                supplier = supplier_input.to_supplier()
                suppliers.append(supplier)
                
            except ValidationError as e:
                logger.warning(f"Validation error for record {idx}: {e}")
                validation_results[f"record_{idx}_error"] = str(e)
            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                validation_results[f"record_{idx}_error"] = str(e)
        
        # Calculate quality score
        quality_score = len(suppliers) / len(supplier_data) if supplier_data else 0.0
        
        # Create data lineage
        data_lineage = DataLineage(
            data_id=f"UPLOAD_{uuid.uuid4().hex[:8].upper()}",
            source_file=path.name,
            user_id=user_id,
            processing_steps=processing_steps,
            validation_results=validation_results,
            quality_score=quality_score,
            row_count=len(supplier_data),
            column_count=0,  # JSON doesn't have columns
            file_size_bytes=file_size
        )
        
        logger.info(f"Parsed JSON file: {len(suppliers)} suppliers from {len(supplier_data)} records")
        return suppliers, data_lineage
    
    def validate_manual_supplier_input(self, supplier_data: Dict[str, Any]) -> Tuple[bool, Union[Supplier, str]]:
        """
        Validate manual supplier input data.
        
        Args:
            supplier_data: Dictionary with supplier information
            
        Returns:
            Tuple of (is_valid, supplier_or_error_message)
        """
        try:
            supplier_input = SupplierDataInput(**supplier_data)
            supplier = supplier_input.to_supplier()
            return True, supplier
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field = error['loc'][0] if error['loc'] else 'unknown'
                message = error['msg']
                error_details.append(f"{field}: {message}")
            return False, "; ".join(error_details)
        except Exception as e:
            return False, str(e)
    
    def validate_manual_risk_input(self, risk_data: Dict[str, Any]) -> Tuple[bool, Union[RiskEventInput, str]]:
        """
        Validate manual risk event input data.
        
        Args:
            risk_data: Dictionary with risk event information
            
        Returns:
            Tuple of (is_valid, risk_event_or_error_message)
        """
        try:
            risk_input = RiskEventInput(**risk_data)
            return True, risk_input
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field = error['loc'][0] if error['loc'] else 'unknown'
                message = error['msg']
                error_details.append(f"{field}: {message}")
            return False, "; ".join(error_details)
        except Exception as e:
            return False, str(e)
    
    def get_file_preview(self, file_path: str, max_rows: int = 5) -> Dict[str, Any]:
        """
        Get a preview of the file contents for user review.
        
        Args:
            file_path: Path to the file
            max_rows: Maximum number of rows to preview
            
        Returns:
            Dictionary with file preview information
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            preview = {
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "extension": extension,
                "preview_rows": [],
                "total_rows": 0,
                "columns": [],
                "detected_mappings": {}
            }
            
            if extension == '.csv':
                delimiter = self._detect_delimiter(file_path)
                df = pd.read_csv(file_path, delimiter=delimiter, nrows=max_rows + 1)
                preview["total_rows"] = len(pd.read_csv(file_path, delimiter=delimiter))
                
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=max_rows + 1)
                preview["total_rows"] = len(pd.read_excel(file_path))
                
            elif extension == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data[:max_rows])
                    preview["total_rows"] = len(data)
                else:
                    df = pd.DataFrame([data])
                    preview["total_rows"] = 1
            else:
                return {"error": f"Unsupported file type: {extension}"}
            
            preview["columns"] = df.columns.tolist()
            preview["preview_rows"] = df.head(max_rows).to_dict('records')
            
            # Detect column mappings
            column_mapping = self._map_columns(df.columns.tolist(), self.supplier_column_mappings)
            preview["detected_mappings"] = column_mapping
            
            return preview
            
        except Exception as e:
            return {"error": f"Failed to preview file: {str(e)}"}


# Convenience functions
def parse_supplier_file(file_path: str, user_id: Optional[str] = None) -> Tuple[List[Supplier], DataLineage]:
    """
    Convenience function to parse supplier data from various file formats.
    
    Args:
        file_path: Path to the file
        user_id: User ID for audit trail
        
    Returns:
        Tuple of (suppliers_list, data_lineage)
    """
    handler = FileInputHandler()
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.csv':
        return handler.parse_csv_file(file_path, user_id)
    elif extension in ['.xlsx', '.xls']:
        return handler.parse_excel_file(file_path, user_id=user_id)
    elif extension == '.json':
        return handler.parse_json_file(file_path, user_id)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def validate_supplier_input(supplier_data: Dict[str, Any]) -> Tuple[bool, Union[Supplier, str]]:
    """
    Convenience function to validate manual supplier input.
    
    Args:
        supplier_data: Dictionary with supplier information
        
    Returns:
        Tuple of (is_valid, supplier_or_error_message)
    """
    handler = FileInputHandler()
    return handler.validate_manual_supplier_input(supplier_data)