"""
File upload and manual data input handlers with validation and lineage tracking.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union, IO
from datetime import datetime
from pathlib import Path
import uuid
import hashlib
from dataclasses import dataclass
from enum import Enum
import json

from pydantic import BaseModel, Field, ValidationError
from models.supplier import Supplier
from models.scraped_data import ScrapedData, DataSource


logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported file types."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    UNKNOWN = "unknown"


class DataLineageEvent(BaseModel):
    """Data lineage tracking event."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str = Field(..., description="Type of lineage event")
    source_file: Optional[str] = Field(None, description="Source file name")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    transformation: Optional[str] = Field(None, description="Transformation applied")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FileValidationResult(BaseModel):
    """File validation result."""
    is_valid: bool
    file_type: FileType
    row_count: int
    column_count: int
    quality_score: float
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data_preview: Dict[str, Any] = Field(default_factory=dict)
    file_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedFileData(BaseModel):
    """Processed file data with lineage."""
    file_id: str
    original_filename: str
    file_type: FileType
    processed_data: List[Dict[str, Any]]
    validation_result: FileValidationResult
    lineage_events: List[DataLineageEvent]
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    data_quality_score: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ManualDataEntry(BaseModel):
    """Manual data entry with validation."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    supplier_id: str
    data_type: str = Field(..., description="Type of manual data entry")
    data_content: Dict[str, Any]
    entered_by: Optional[str] = Field(None, description="User who entered the data")
    entry_timestamp: datetime = Field(default_factory=datetime.now)
    validation_status: str = Field(default="pending")
    quality_score: float = Field(default=0.5, ge=0, le=1)
    notes: Optional[str] = Field(None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FileUploadHandler:
    """
    File upload handler with CSV/Excel parsing, validation, and lineage tracking.
    """
    
    def __init__(self):
        """Initialize the file upload handler."""
        self.supported_extensions = {
            '.csv': FileType.CSV,
            '.xlsx': FileType.EXCEL,
            '.xls': FileType.EXCEL,
            '.json': FileType.JSON
        }
        
        # Expected columns for different data types
        self.expected_columns = {
            'supplier': ['id', 'name', 'country', 'industry'],
            'financial': ['supplier_id', 'revenue', 'profit', 'debt'],
            'risk_event': ['supplier_id', 'event_type', 'severity', 'description'],
            'general': []  # No specific requirements
        }
    
    def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content."""
        return hashlib.sha256(file_content).hexdigest()
    
    def _detect_file_type(self, filename: str) -> FileType:
        """Detect file type from filename extension."""
        path = Path(filename)
        extension = path.suffix.lower()
        return self.supported_extensions.get(extension, FileType.UNKNOWN)
    
    def _validate_data_quality(self, df: pd.DataFrame, data_type: str = 'general') -> tuple[float, List[str], List[str]]:
        """
        Validate data quality and return score, errors, and warnings.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data for specific validation rules
            
        Returns:
            Tuple of (quality_score, errors, warnings)
        """
        errors = []
        warnings = []
        quality_score = 1.0
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("File contains no data")
            return 0.0, errors, warnings
        
        # Check for missing values
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percentage > 50:
            errors.append(f"Too many missing values: {missing_percentage:.1f}%")
            quality_score -= 0.3
        elif missing_percentage > 20:
            warnings.append(f"High percentage of missing values: {missing_percentage:.1f}%")
            quality_score -= 0.1
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")
            quality_score -= 0.05
        
        # Data type specific validations
        expected_cols = self.expected_columns.get(data_type, [])
        if expected_cols:
            missing_cols = set(expected_cols) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing required columns: {list(missing_cols)}")
                quality_score -= 0.2
        
        # Check for reasonable data ranges
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].min() < 0 and col in ['revenue', 'profit', 'employee_count']:
                warnings.append(f"Negative values found in {col}")
                quality_score -= 0.05
        
        return max(quality_score, 0.0), errors, warnings
    
    def _create_data_preview(self, df: pd.DataFrame, max_rows: int = 5) -> Dict[str, Any]:
        """Create a preview of the data for validation result."""
        preview = {
            "columns": list(df.columns),
            "sample_rows": df.head(max_rows).to_dict('records'),
            "data_types": df.dtypes.astype(str).to_dict(),
            "summary_stats": {}
        }
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            preview["summary_stats"] = df[numeric_cols].describe().to_dict()
        
        return preview
    
    async def validate_file(self, file_path: str, data_type: str = 'general') -> FileValidationResult:
        """
        Validate uploaded file and return validation result.
        
        Args:
            file_path: Path to the uploaded file
            data_type: Type of data for specific validation
            
        Returns:
            FileValidationResult object
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return FileValidationResult(
                    is_valid=False,
                    file_type=FileType.UNKNOWN,
                    row_count=0,
                    column_count=0,
                    quality_score=0.0,
                    errors=[f"File not found: {file_path}"],
                    file_hash=""
                )
            
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_content = f.read()
            file_hash = self._calculate_file_hash(file_content)
            
            # Detect file type
            file_type = self._detect_file_type(path.name)
            
            if file_type == FileType.UNKNOWN:
                return FileValidationResult(
                    is_valid=False,
                    file_type=file_type,
                    row_count=0,
                    column_count=0,
                    quality_score=0.0,
                    errors=[f"Unsupported file type: {path.suffix}"],
                    file_hash=file_hash
                )
            
            # Load data based on file type
            df = None
            if file_type == FileType.CSV:
                df = pd.read_csv(file_path)
            elif file_type == FileType.EXCEL:
                df = pd.read_excel(file_path)
            elif file_type == FileType.JSON:
                df = pd.read_json(file_path)
            
            if df is None:
                return FileValidationResult(
                    is_valid=False,
                    file_type=file_type,
                    row_count=0,
                    column_count=0,
                    quality_score=0.0,
                    errors=["Failed to load file data"],
                    file_hash=file_hash
                )
            
            # Validate data quality
            quality_score, errors, warnings = self._validate_data_quality(df, data_type)
            
            # Create data preview
            data_preview = self._create_data_preview(df)
            
            return FileValidationResult(
                is_valid=len(errors) == 0,
                file_type=file_type,
                row_count=len(df),
                column_count=len(df.columns),
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                data_preview=data_preview,
                file_hash=file_hash,
                metadata={
                    "file_size": len(file_content),
                    "original_filename": path.name
                }
            )
            
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {str(e)}")
            return FileValidationResult(
                is_valid=False,
                file_type=FileType.UNKNOWN,
                row_count=0,
                column_count=0,
                quality_score=0.0,
                errors=[f"Validation error: {str(e)}"],
                file_hash=""
            )
    
    async def process_file(self, file_path: str, data_type: str = 'general', 
                          user_id: Optional[str] = None) -> ProcessedFileData:
        """
        Process uploaded file with validation and lineage tracking.
        
        Args:
            file_path: Path to the uploaded file
            data_type: Type of data for processing
            user_id: ID of user who uploaded the file
            
        Returns:
            ProcessedFileData object
        """
        file_id = str(uuid.uuid4())
        path = Path(file_path)
        
        # Create initial lineage event
        lineage_events = [
            DataLineageEvent(
                event_type="file_upload",
                source_file=path.name,
                user_id=user_id,
                metadata={
                    "file_path": str(path),
                    "data_type": data_type
                }
            )
        ]
        
        try:
            # Validate file
            validation_result = await self.validate_file(file_path, data_type)
            
            lineage_events.append(
                DataLineageEvent(
                    event_type="file_validation",
                    source_file=path.name,
                    user_id=user_id,
                    metadata={
                        "is_valid": validation_result.is_valid,
                        "quality_score": validation_result.quality_score,
                        "error_count": len(validation_result.errors)
                    }
                )
            )
            
            if not validation_result.is_valid:
                return ProcessedFileData(
                    file_id=file_id,
                    original_filename=path.name,
                    file_type=validation_result.file_type,
                    processed_data=[],
                    validation_result=validation_result,
                    lineage_events=lineage_events,
                    data_quality_score=validation_result.quality_score
                )
            
            # Load and process data
            df = None
            if validation_result.file_type == FileType.CSV:
                df = pd.read_csv(file_path)
            elif validation_result.file_type == FileType.EXCEL:
                df = pd.read_excel(file_path)
            elif validation_result.file_type == FileType.JSON:
                df = pd.read_json(file_path)
            
            # Clean and transform data
            df_cleaned = self._clean_data(df)
            
            lineage_events.append(
                DataLineageEvent(
                    event_type="data_cleaning",
                    source_file=path.name,
                    user_id=user_id,
                    transformation="remove_nulls_and_duplicates",
                    metadata={
                        "original_rows": len(df),
                        "cleaned_rows": len(df_cleaned),
                        "rows_removed": len(df) - len(df_cleaned)
                    }
                )
            )
            
            # Convert to list of dictionaries
            processed_data = df_cleaned.to_dict('records')
            
            lineage_events.append(
                DataLineageEvent(
                    event_type="data_processing_complete",
                    source_file=path.name,
                    user_id=user_id,
                    metadata={
                        "final_record_count": len(processed_data)
                    }
                )
            )
            
            return ProcessedFileData(
                file_id=file_id,
                original_filename=path.name,
                file_type=validation_result.file_type,
                processed_data=processed_data,
                validation_result=validation_result,
                lineage_events=lineage_events,
                data_quality_score=validation_result.quality_score
            )
            
        except Exception as e:
            logger.error(f"File processing failed for {file_path}: {str(e)}")
            
            lineage_events.append(
                DataLineageEvent(
                    event_type="processing_error",
                    source_file=path.name,
                    user_id=user_id,
                    metadata={
                        "error": str(e)
                    }
                )
            )
            
            return ProcessedFileData(
                file_id=file_id,
                original_filename=path.name,
                file_type=FileType.UNKNOWN,
                processed_data=[],
                validation_result=FileValidationResult(
                    is_valid=False,
                    file_type=FileType.UNKNOWN,
                    row_count=0,
                    column_count=0,
                    quality_score=0.0,
                    errors=[f"Processing error: {str(e)}"],
                    file_hash=""
                ),
                lineage_events=lineage_events,
                data_quality_score=0.0
            )
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing nulls and duplicates.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Remove completely empty rows
        df_cleaned = df.dropna(how='all')
        
        # Remove duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()
        
        # Fill numeric NaN values with 0 for specific columns
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col in ['revenue', 'profit', 'employee_count']:
                df_cleaned[col] = df_cleaned[col].fillna(0)
        
        return df_cleaned
    
    def convert_to_suppliers(self, processed_data: ProcessedFileData) -> List[Supplier]:
        """
        Convert processed file data to Supplier objects.
        
        Args:
            processed_data: ProcessedFileData object
            
        Returns:
            List of Supplier objects
        """
        suppliers = []
        
        for record in processed_data.processed_data:
            try:
                # Map common column variations
                supplier_data = {}
                
                # ID mapping
                for id_col in ['id', 'supplier_id', 'ID', 'Supplier_ID']:
                    if id_col in record:
                        supplier_data['id'] = str(record[id_col])
                        break
                
                # Name mapping
                for name_col in ['name', 'company_name', 'supplier_name', 'Name']:
                    if name_col in record:
                        supplier_data['name'] = str(record[name_col])
                        break
                
                # Country mapping
                for country_col in ['country', 'Country', 'location']:
                    if country_col in record:
                        supplier_data['country'] = str(record[country_col])
                        break
                
                # Industry mapping
                for industry_col in ['industry', 'Industry', 'sector', 'business_type']:
                    if industry_col in record:
                        supplier_data['industry'] = str(record[industry_col])
                        break
                
                # Set defaults for required fields
                if 'id' not in supplier_data:
                    supplier_data['id'] = f"SUP_{uuid.uuid4().hex[:8].upper()}"
                if 'name' not in supplier_data:
                    supplier_data['name'] = "Unknown Supplier"
                if 'country' not in supplier_data:
                    supplier_data['country'] = "Unknown"
                if 'industry' not in supplier_data:
                    supplier_data['industry'] = "Unknown"
                
                # Add region based on country (simplified mapping)
                region_mapping = {
                    "United States": "North America",
                    "Canada": "North America",
                    "Mexico": "North America",
                    "Germany": "Europe",
                    "France": "Europe",
                    "United Kingdom": "Europe",
                    "Italy": "Europe",
                    "Spain": "Europe",
                    "China": "Asia",
                    "Japan": "Asia",
                    "South Korea": "Asia",
                    "India": "Asia",
                    "Australia": "Oceania",
                    "Brazil": "South America"
                }
                supplier_data['region'] = region_mapping.get(supplier_data['country'], "Unknown")
                
                # Optional fields
                for field, columns in {
                    'website': ['website', 'url', 'Website'],
                    'annual_revenue': ['revenue', 'annual_revenue', 'Revenue'],
                    'employee_count': ['employees', 'employee_count', 'staff_count'],
                    'financial_health_score': ['financial_score', 'health_score', 'rating']
                }.items():
                    for col in columns:
                        if col in record and record[col] is not None:
                            if field in ['annual_revenue', 'employee_count', 'financial_health_score']:
                                try:
                                    supplier_data[field] = float(record[col])
                                except (ValueError, TypeError):
                                    pass
                            else:
                                supplier_data[field] = str(record[col])
                            break
                
                supplier = Supplier(**supplier_data)
                suppliers.append(supplier)
                
            except ValidationError as e:
                logger.warning(f"Failed to create supplier from record: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error creating supplier: {e}")
                continue
        
        return suppliers


class ManualDataInputHandler:
    """
    Manual data input handler with Pydantic validation.
    """
    
    def __init__(self):
        """Initialize the manual data input handler."""
        self.data_types = {
            'supplier_info': {
                'required_fields': ['name', 'country', 'industry'],
                'optional_fields': ['website', 'annual_revenue', 'employee_count']
            },
            'financial_data': {
                'required_fields': ['supplier_id', 'revenue'],
                'optional_fields': ['profit', 'debt', 'financial_health_score']
            },
            'risk_event': {
                'required_fields': ['supplier_id', 'event_type', 'severity'],
                'optional_fields': ['description', 'probability', 'impact_score']
            }
        }
    
    def validate_manual_entry(self, data_type: str, data_content: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate manual data entry.
        
        Args:
            data_type: Type of data being entered
            data_content: Data content to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        if data_type not in self.data_types:
            errors.append(f"Unknown data type: {data_type}")
            return False, errors
        
        type_config = self.data_types[data_type]
        
        # Check required fields
        for field in type_config['required_fields']:
            if field not in data_content or not data_content[field]:
                errors.append(f"Required field missing: {field}")
        
        # Validate data types for specific fields
        if data_type == 'financial_data':
            for field in ['revenue', 'profit', 'debt']:
                if field in data_content:
                    try:
                        float(data_content[field])
                    except (ValueError, TypeError):
                        errors.append(f"Invalid numeric value for {field}")
        
        return len(errors) == 0, errors
    
    def create_manual_entry(self, supplier_id: str, data_type: str, 
                          data_content: Dict[str, Any], entered_by: Optional[str] = None,
                          notes: Optional[str] = None) -> ManualDataEntry:
        """
        Create a manual data entry with validation.
        
        Args:
            supplier_id: ID of the supplier
            data_type: Type of data being entered
            data_content: Data content
            entered_by: User who entered the data
            notes: Optional notes
            
        Returns:
            ManualDataEntry object
        """
        is_valid, errors = self.validate_manual_entry(data_type, data_content)
        
        # Calculate quality score based on completeness and validity
        quality_score = 0.5  # Base score
        if is_valid:
            quality_score += 0.3
        
        type_config = self.data_types.get(data_type, {})
        total_possible_fields = len(type_config.get('required_fields', [])) + len(type_config.get('optional_fields', []))
        if total_possible_fields > 0:
            provided_fields = len([f for f in data_content.keys() if data_content[f] is not None])
            completeness = provided_fields / total_possible_fields
            quality_score += completeness * 0.2
        
        return ManualDataEntry(
            supplier_id=supplier_id,
            data_type=data_type,
            data_content=data_content,
            entered_by=entered_by,
            validation_status="valid" if is_valid else "invalid",
            quality_score=min(quality_score, 1.0),
            notes=notes
        )
    
    def convert_to_scraped_data(self, manual_entry: ManualDataEntry) -> ScrapedData:
        """
        Convert manual data entry to ScrapedData format.
        
        Args:
            manual_entry: ManualDataEntry object
            
        Returns:
            ScrapedData object
        """
        content = json.dumps(manual_entry.data_content, indent=2)
        
        return ScrapedData(
            data_id=f"MANUAL_{manual_entry.entry_id[:8].upper()}",
            supplier_id=manual_entry.supplier_id,
            source_url="https://manual-entry.local",
            content=content,
            extraction_method="manual_input",
            relevance_score=1.0,  # Manual entries are always relevant
            quality_score=manual_entry.quality_score,
            data_source=DataSource.MANUAL_INPUT,
            metadata={
                "entry_id": manual_entry.entry_id,
                "data_type": manual_entry.data_type,
                "entered_by": manual_entry.entered_by,
                "entry_timestamp": manual_entry.entry_timestamp.isoformat(),
                "validation_status": manual_entry.validation_status,
                "notes": manual_entry.notes
            },
            processing_status="processed"
        )


# Convenience functions
async def process_uploaded_file(file_path: str, data_type: str = 'general', 
                               user_id: Optional[str] = None) -> ProcessedFileData:
    """
    Convenience function to process an uploaded file.
    
    Args:
        file_path: Path to the uploaded file
        data_type: Type of data for processing
        user_id: ID of user who uploaded the file
        
    Returns:
        ProcessedFileData object
    """
    handler = FileUploadHandler()
    return await handler.process_file(file_path, data_type, user_id)


def create_manual_data_entry(supplier_id: str, data_type: str, 
                           data_content: Dict[str, Any], entered_by: Optional[str] = None,
                           notes: Optional[str] = None) -> ManualDataEntry:
    """
    Convenience function to create a manual data entry.
    
    Args:
        supplier_id: ID of the supplier
        data_type: Type of data being entered
        data_content: Data content
        entered_by: User who entered the data
        notes: Optional notes
        
    Returns:
        ManualDataEntry object
    """
    handler = ManualDataInputHandler()
    return handler.create_manual_entry(supplier_id, data_type, data_content, entered_by, notes)