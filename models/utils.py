"""
Utility functions for data conversion between pandas and Pydantic models.
"""

import pandas as pd
from typing import List, Type, TypeVar, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def pandas_to_pydantic(df: pd.DataFrame, model_class: Type[T], 
                      id_column: Optional[str] = None) -> List[T]:
    """
    Convert pandas DataFrame to list of Pydantic model instances.
    
    Args:
        df: Input DataFrame
        model_class: Pydantic model class to convert to
        id_column: Optional column name to use as unique identifier
        
    Returns:
        List of validated Pydantic model instances
        
    Raises:
        ValidationError: If data doesn't match model schema
    """
    if df.empty:
        return []
    
    # Convert DataFrame to list of dictionaries
    records = df.to_dict('records')
    validated_models = []
    validation_errors = []
    
    for i, record in enumerate(records):
        try:
            # Clean None values and convert numpy types
            cleaned_record = _clean_record_for_pydantic(record)
            
            # Add row identifier if specified
            if id_column and id_column in cleaned_record:
                cleaned_record['id'] = str(cleaned_record[id_column])
            elif 'id' not in cleaned_record:
                cleaned_record['id'] = f"AUTO_{i+1:04d}"
            
            # Create and validate model instance
            model_instance = model_class(**cleaned_record)
            validated_models.append(model_instance)
            
        except ValidationError as e:
            error_msg = f"Row {i}: {str(e)}"
            validation_errors.append(error_msg)
            logger.warning(f"Validation error for row {i}: {e}")
        except Exception as e:
            error_msg = f"Row {i}: Unexpected error - {str(e)}"
            validation_errors.append(error_msg)
            logger.error(f"Unexpected error for row {i}: {e}")
    
    if validation_errors:
        logger.warning(f"Converted {len(validated_models)} records with {len(validation_errors)} errors")
    
    return validated_models


def pydantic_to_pandas(models: List[BaseModel], 
                      exclude_fields: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert list of Pydantic model instances to pandas DataFrame.
    
    Args:
        models: List of Pydantic model instances
        exclude_fields: Optional list of field names to exclude
        
    Returns:
        pandas DataFrame with model data
    """
    if not models:
        return pd.DataFrame()
    
    exclude_fields = exclude_fields or []
    
    # Convert models to dictionaries
    records = []
    for model in models:
        record = model.dict(exclude=set(exclude_fields))
        # Convert datetime objects to strings for pandas compatibility
        record = _prepare_record_for_pandas(record)
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Optimize data types
    df = _optimize_dataframe_dtypes(df)
    
    return df


def validate_dataframe_schema(df: pd.DataFrame, model_class: Type[T]) -> Dict[str, Any]:
    """
    Validate DataFrame schema against Pydantic model.
    
    Args:
        df: DataFrame to validate
        model_class: Pydantic model class for validation
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "is_valid": True,
        "missing_columns": [],
        "extra_columns": [],
        "type_mismatches": [],
        "sample_errors": []
    }
    
    # Get model fields
    model_fields = model_class.__fields__
    required_fields = [name for name, field in model_fields.items() if field.required]
    
    # Check for missing required columns
    df_columns = set(df.columns)
    model_columns = set(model_fields.keys())
    
    validation_result["missing_columns"] = [
        col for col in required_fields if col not in df_columns
    ]
    validation_result["extra_columns"] = list(df_columns - model_columns)
    
    # Test conversion with sample data
    if not df.empty and not validation_result["missing_columns"]:
        try:
            sample_size = min(5, len(df))
            sample_models = pandas_to_pydantic(df.head(sample_size), model_class)
            if len(sample_models) < sample_size:
                validation_result["sample_errors"].append(
                    f"Only {len(sample_models)}/{sample_size} sample records converted successfully"
                )
        except Exception as e:
            validation_result["sample_errors"].append(str(e))
    
    # Set overall validity
    validation_result["is_valid"] = (
        not validation_result["missing_columns"] and
        not validation_result["sample_errors"]
    )
    
    return validation_result


def _clean_record_for_pydantic(record: Dict[str, Any]) -> Dict[str, Any]:
    """Clean record data for Pydantic validation."""
    cleaned = {}
    
    for key, value in record.items():
        # Skip None values for optional fields
        if pd.isna(value):
            continue
            
        # Convert numpy types to Python types
        if hasattr(value, 'item'):  # numpy scalar
            value = value.item()
        elif isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
            value = float(value) if pd.notna(value) else None
            
        # Clean string values
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                continue
                
        cleaned[key] = value
    
    return cleaned


def _prepare_record_for_pandas(record: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare record for pandas DataFrame creation."""
    prepared = {}
    
    for key, value in record.items():
        if isinstance(value, datetime):
            prepared[key] = value.isoformat()
        elif isinstance(value, (list, dict)):
            # Convert complex types to strings for pandas
            prepared[key] = str(value)
        else:
            prepared[key] = value
            
    return prepared


def _optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for memory efficiency."""
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        # Optimize numeric columns
        if col_type in ['int64', 'float64']:
            if col_type == 'int64':
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            else:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize string columns
        elif col_type == 'object':
            # Check if it's actually categorical
            unique_ratio = optimized_df[col].nunique() / len(optimized_df[col])
            if unique_ratio < 0.5:  # Less than 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
    
    return optimized_df


def create_sample_data(model_class: Type[T], count: int = 10) -> List[T]:
    """
    Create sample data instances for testing purposes.
    
    Args:
        model_class: Pydantic model class
        count: Number of sample instances to create
        
    Returns:
        List of sample model instances
    """
    # This is a basic implementation - in practice, you'd want more sophisticated
    # sample data generation based on the model schema
    samples = []
    
    try:
        # Get example from model config if available
        example = getattr(model_class.Config, 'schema_extra', {}).get('example', {})
        
        for i in range(count):
            sample_data = example.copy()
            # Modify ID to make it unique
            if 'id' in sample_data:
                sample_data['id'] = f"{sample_data['id']}_{i+1:03d}"
            elif hasattr(model_class, 'id'):
                sample_data['id'] = f"SAMPLE_{i+1:03d}"
                
            samples.append(model_class(**sample_data))
            
    except Exception as e:
        logger.error(f"Failed to create sample data for {model_class.__name__}: {e}")
    
    return samples