"""
Flexible AI Agent for Dynamic Document Parsing and Data Extraction
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
from pathlib import Path
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

class FlexibleParserAgent:
    """AI agent that can parse any document format and extract structured data."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        
    def parse_document(self, file_path: str = None, file_content: str = None, file_type: str = "", user_intent: str = "") -> Dict[str, Any]:
        """Parse any document format using Docling + AI to extract supplier data."""
        
        # Use Docling for document parsing if available and file_path provided
        if DOCLING_AVAILABLE and file_path and file_type.lower() in ['pdf', 'docx', 'doc', 'pptx']:
            try:
                converter = DocumentConverter()
                result = converter.convert(file_path)
                file_content = result.document.export_to_markdown()
                file_type = "markdown"
            except Exception as e:
                pass  # Fall back to original content
        
        # Use original content if Docling not available or failed
        if not file_content:
            return {"suppliers": [], "metadata": {"error": "No content to parse"}}
        
        prompt = f"""
        Extract supplier information from this {file_type} content.
        
        User Intent: {user_intent}
        
        Content:
        {file_content[:5000]}
        
        Return JSON:
        {{
            "suppliers": [
                {{
                    "id": "unique_id",
                    "name": "company_name", 
                    "country": "country",
                    "region": "region",
                    "industry": "industry",
                    "tier": "tier_1|tier_2|tier_3",
                    "criticality_score": 0-100
                }}
            ],
            "metadata": {{
                "total_records": "number",
                "data_quality": "0-1",
                "extraction_confidence": "0-1",
                "detected_format": "format_description"
            }}
        }}
        """
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                return self._fallback_parse(file_content, file_type)
        else:
            return self._fallback_parse(file_content, file_type)
    
    def _fallback_parse(self, content: str, file_type: str) -> Dict[str, Any]:
        """Fallback parsing without AI when API unavailable."""
        
        if file_type.lower() == 'csv':
            return self._parse_csv_fallback(content)
        elif file_type.lower() in ['json', 'jsonl']:
            return self._parse_json_fallback(content)
        else:
            return self._parse_text_fallback(content)
    
    def _parse_csv_fallback(self, content: str) -> Dict[str, Any]:
        """Parse CSV with flexible column mapping."""
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(content))
            
            # Flexible column mapping
            column_map = self._map_columns(df.columns.tolist())
            suppliers = []
            
            for idx, row in df.iterrows():
                supplier = {
                    "id": self._get_value(row, column_map.get('id', []), f"SUP{idx+1:03d}"),
                    "name": self._get_value(row, column_map.get('name', []), f"Supplier {idx+1}"),
                    "country": self._get_value(row, column_map.get('country', []), "Unknown"),
                    "region": self._infer_region(self._get_value(row, column_map.get('country', []), "Unknown")),
                    "industry": self._get_value(row, column_map.get('industry', []), "General"),
                    "tier": self._get_value(row, column_map.get('tier', []), "tier_2"),
                    "criticality_score": float(self._get_value(row, column_map.get('criticality', []), 50)),
                    "website": self._get_value(row, column_map.get('website', []), None),
                    "annual_revenue": self._safe_float(self._get_value(row, column_map.get('revenue', []), None)),
                    "employee_count": self._safe_int(self._get_value(row, column_map.get('employees', []), None)),
                    "financial_health_score": self._safe_float(self._get_value(row, column_map.get('financial', []), None))
                }
                suppliers.append(supplier)
            
            return {
                "suppliers": suppliers,
                "metadata": {
                    "total_records": len(suppliers),
                    "data_quality": 0.8,
                    "extraction_confidence": 0.9,
                    "detected_format": "CSV with flexible mapping"
                }
            }
        except Exception as e:
            return {"suppliers": [], "metadata": {"error": str(e)}}
    
    def _map_columns(self, columns: List[str]) -> Dict[str, List[str]]:
        """Map various column names to standard fields."""
        mapping = {
            'id': ['id', 'supplier_id', 'code', 'identifier'],
            'name': ['name', 'company', 'supplier_name', 'organization'],
            'country': ['country', 'location', 'nation'],
            'industry': ['industry', 'sector', 'business', 'category'],
            'tier': ['tier', 'level', 'class', 'grade'],
            'criticality': ['criticality', 'importance', 'priority', 'critical'],
            'website': ['website', 'url', 'web', 'site'],
            'revenue': ['revenue', 'sales', 'turnover', 'income'],
            'employees': ['employees', 'staff', 'workforce', 'headcount'],
            'financial': ['financial', 'health', 'score', 'rating']
        }
        
        result = {}
        for field, keywords in mapping.items():
            result[field] = [col for col in columns 
                           if any(keyword.lower() in col.lower() for keyword in keywords)]
        
        return result
    
    def _get_value(self, row, possible_columns: List[str], default: Any) -> Any:
        """Get value from row using possible column names."""
        for col in possible_columns:
            if col in row and pd.notna(row[col]):
                return row[col]
        return default
    
    def _infer_region(self, country: str) -> str:
        """Infer region from country name."""
        country = country.lower()
        
        if any(c in country for c in ['germany', 'france', 'uk', 'italy', 'spain', 'netherlands']):
            return "Europe"
        elif any(c in country for c in ['china', 'japan', 'korea', 'taiwan', 'singapore', 'india']):
            return "Asia Pacific"
        elif any(c in country for c in ['usa', 'canada', 'mexico', 'united states']):
            return "North America"
        elif any(c in country for c in ['brazil', 'argentina', 'chile', 'colombia']):
            return "South America"
        else:
            return "Other"
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert to float."""
        try:
            return float(value) if value is not None else None
        except:
            return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert to int."""
        try:
            return int(value) if value is not None else None
        except:
            return None
    
    def _parse_json_fallback(self, content: str) -> Dict[str, Any]:
        """Parse JSON with flexible structure."""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                suppliers = data
            elif isinstance(data, dict) and 'suppliers' in data:
                suppliers = data['suppliers']
            else:
                suppliers = [data]
            
            return {
                "suppliers": suppliers,
                "metadata": {
                    "total_records": len(suppliers),
                    "data_quality": 0.9,
                    "extraction_confidence": 0.95,
                    "detected_format": "JSON"
                }
            }
        except Exception as e:
            return {"suppliers": [], "metadata": {"error": str(e)}}
    
    def _parse_text_fallback(self, content: str) -> Dict[str, Any]:
        """Parse unstructured text."""
        lines = content.split('\n')
        suppliers = []
        
        # Simple text parsing - look for company names and info
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                suppliers.append({
                    "id": f"SUP{i+1:03d}",
                    "name": line.strip()[:100],
                    "country": "Unknown",
                    "region": "Other",
                    "industry": "General",
                    "tier": "tier_2",
                    "criticality_score": 50.0
                })
        
        return {
            "suppliers": suppliers[:50],  # Limit to 50
            "metadata": {
                "total_records": len(suppliers),
                "data_quality": 0.5,
                "extraction_confidence": 0.6,
                "detected_format": "Unstructured text"
            }
        }