"""
Agile AI-Powered Workflow - Flexible and adaptive supply chain analysis
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from agents.flexible_parser_agent import FlexibleParserAgent
from agents.dynamic_analysis_agent import DynamicAnalysisAgent
from models.supplier import Supplier
from models.analysis_result import AnalysisResult
from models.risk_event import RiskEvent, RiskLevel
from models.ml_prediction import MLPrediction

class AgileWorkflow:
    """Fully flexible AI-powered workflow that adapts to any input format."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.parser_agent = FlexibleParserAgent(openai_api_key)
        self.analysis_agent = DynamicAnalysisAgent(openai_api_key)
        self.workflow_id = f"AGILE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def process_any_input(self, 
                               input_data: Any, 
                               input_type: str = "auto",
                               user_context: str = "") -> AnalysisResult:
        """Process any type of input and return comprehensive analysis."""
        
        try:
            # Step 1: Parse input flexibly
            parsed_data = await self._parse_input(input_data, input_type, user_context)
            
            # Step 2: Create supplier objects
            suppliers = self._create_suppliers(parsed_data)
            
            # Step 3: Perform AI analysis
            analysis_results = await self._perform_analysis(suppliers, user_context)
            
            # Step 4: Generate final result
            return self._create_analysis_result(suppliers, analysis_results, parsed_data)
            
        except Exception as e:
            # Graceful degradation
            return self._create_fallback_result(str(e))
    
    async def _parse_input(self, input_data: Any, input_type: str, context: str) -> Dict[str, Any]:
        """Parse any input format using AI agent."""
        
        if input_type == "auto":
            input_type = self._detect_input_type(input_data)
        
        if isinstance(input_data, str):
            # File content or raw text
            return self.parser_agent.parse_document(input_data, input_type, context)
        elif isinstance(input_data, dict):
            # JSON-like data
            return {"suppliers": [input_data], "metadata": {"detected_format": "dict"}}
        elif isinstance(input_data, list):
            # List of suppliers
            return {"suppliers": input_data, "metadata": {"detected_format": "list"}}
        else:
            # Convert to string and parse
            return self.parser_agent.parse_document(str(input_data), "text", context)
    
    def _detect_input_type(self, data: Any) -> str:
        """Auto-detect input type."""
        if isinstance(data, str):
            if data.strip().startswith('{') or data.strip().startswith('['):
                return "json"
            elif ',' in data and '\n' in data:
                return "csv"
            else:
                return "text"
        elif isinstance(data, (dict, list)):
            return "json"
        else:
            return "text"
    
    def _create_suppliers(self, parsed_data: Dict[str, Any]) -> List[Supplier]:
        """Create Supplier objects from parsed data."""
        suppliers = []
        
        for supplier_data in parsed_data.get("suppliers", []):
            try:
                # Fill in missing required fields with defaults
                supplier_dict = {
                    "id": supplier_data.get("id", f"SUP{len(suppliers)+1:03d}"),
                    "name": supplier_data.get("name", "Unknown Supplier"),
                    "country": supplier_data.get("country", "Unknown"),
                    "region": supplier_data.get("region", "Other"),
                    "industry": supplier_data.get("industry", "General"),
                    "tier": supplier_data.get("tier", "tier_2"),
                    "criticality_score": float(supplier_data.get("criticality_score", 50.0)),
                    "website": supplier_data.get("website"),
                    "annual_revenue": supplier_data.get("annual_revenue"),
                    "employee_count": supplier_data.get("employee_count"),
                    "financial_health_score": supplier_data.get("financial_health_score")
                }
                
                supplier = Supplier(**supplier_dict)
                suppliers.append(supplier)
                
            except Exception as e:
                # Skip invalid suppliers but continue processing
                continue
        
        return suppliers
    
    async def _perform_analysis(self, suppliers: List[Supplier], context: str) -> Dict[str, Any]:
        """Perform comprehensive analysis using AI agent."""
        return self.analysis_agent.analyze_suppliers(suppliers, context)
    
    def _create_analysis_result(self, 
                               suppliers: List[Supplier], 
                               analysis: Dict[str, Any],
                               parsed_metadata: Dict[str, Any]) -> AnalysisResult:
        """Create comprehensive analysis result."""
        
        # Create risk events
        risk_events = []
        for event_data in analysis.get("risk_events", []):
            try:
                risk_event = RiskEvent(
                    event_id=event_data["event_id"],
                    supplier_id=event_data["supplier_id"],
                    event_type=event_data["event_type"],
                    severity=RiskLevel(event_data["severity"]),
                    probability=event_data["probability"],
                    impact_score=event_data["impact_score"],
                    confidence_level=event_data["confidence_level"],
                    description=event_data["description"],
                    evidence=event_data.get("evidence", []),
                    geographic_scope=event_data.get("geographic_scope"),
                    predicted_timeline=event_data.get("predicted_timeline"),
                    mitigation_actions=event_data.get("mitigation_actions", [])
                )
                risk_events.append(risk_event)
            except Exception:
                continue
        
        # Create ML predictions
        ml_predictions = []
        for pred_data in analysis.get("ml_predictions", []):
            try:
                prediction = MLPrediction(
                    prediction_id=pred_data["prediction_id"],
                    supplier_id=pred_data["supplier_id"],
                    model_version="AI_Agent_v1.0",
                    prediction_type=pred_data["prediction_type"],
                    prediction=pred_data["prediction"],
                    confidence=pred_data["confidence"],
                    feature_importance=pred_data.get("feature_importance", {})
                )
                ml_predictions.append(prediction)
            except Exception:
                continue
        
        # Create analysis result
        return AnalysisResult(
            analysis_id=self.workflow_id,
            suppliers=suppliers,
            risk_events=risk_events,
            ml_predictions=ml_predictions,
            overall_risk_score=analysis.get("overall_risk_score", 50.0),
            data_quality_score=parsed_metadata.get("metadata", {}).get("data_quality", 0.8),
            recommendations=analysis.get("recommendations", []),
            workflow_status="completed",
            processing_time=1000.0,  # Placeholder
            generated_at=datetime.now()
        )
    
    def _create_fallback_result(self, error_message: str) -> AnalysisResult:
        """Create fallback result when processing fails."""
        return AnalysisResult(
            analysis_id=f"FALLBACK_{self.workflow_id}",
            suppliers=[],
            risk_events=[],
            ml_predictions=[],
            overall_risk_score=0.0,
            data_quality_score=0.0,
            recommendations=[f"Processing failed: {error_message}"],
            workflow_status="failed",
            processing_time=0.0,
            generated_at=datetime.now()
        )
    
    def process_file_upload(self, file_content: str, filename: str, user_intent: str = "") -> AnalysisResult:
        """Synchronous wrapper for file upload processing."""
        file_extension = Path(filename).suffix.lower()
        
        # Map file extensions to types
        type_mapping = {
            '.csv': 'csv',
            '.json': 'json',
            '.txt': 'text',
            '.xlsx': 'excel',
            '.xml': 'xml',
            '.pdf': 'pdf'
        }
        
        input_type = type_mapping.get(file_extension, 'text')
        
        # Run async workflow
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.process_any_input(file_content, input_type, user_intent)
            )
            return result
        finally:
            loop.close()
    
    def process_manual_input(self, suppliers_data: List[Dict[str, Any]]) -> AnalysisResult:
        """Process manually entered supplier data."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.process_any_input(suppliers_data, "json", "Manual data entry")
            )
            return result
        finally:
            loop.close()
    
    def get_workflow_capabilities(self) -> Dict[str, Any]:
        """Return current workflow capabilities."""
        return {
            "supported_formats": [
                "CSV (any column structure)",
                "JSON (any schema)",
                "Excel files",
                "Plain text",
                "XML documents",
                "PDF documents (with text extraction)",
                "Manual data entry",
                "API responses"
            ],
            "analysis_features": [
                "AI-powered risk assessment",
                "Dynamic supplier scoring",
                "Geopolitical risk analysis",
                "Financial health evaluation",
                "Industry-specific insights",
                "Predictive risk modeling",
                "Mitigation recommendations"
            ],
            "flexibility_features": [
                "Auto-format detection",
                "Intelligent field mapping",
                "Missing data inference",
                "Adaptive analysis based on available data",
                "Graceful degradation",
                "Context-aware processing"
            ],
            "ai_capabilities": [
                "Natural language processing",
                "Document understanding",
                "Risk pattern recognition",
                "Strategic insight generation",
                "Automated report generation"
            ]
        }