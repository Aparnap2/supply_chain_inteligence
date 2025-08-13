"""
Dynamic Analysis Agent - AI-powered flexible risk analysis
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
from models.supplier import Supplier
from models.risk_event import RiskEvent, RiskLevel

class DynamicAnalysisAgent:
    """AI agent that performs flexible risk analysis based on available data."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
    
    def analyze_suppliers(self, suppliers: List[Supplier], context: str = "") -> Dict[str, Any]:
        """Perform comprehensive risk analysis on suppliers using AI."""
        
        # Prepare supplier data for analysis
        supplier_data = [
            {
                "id": s.id,
                "name": s.name,
                "country": s.country,
                "region": s.region,
                "industry": s.industry,
                "tier": s.tier,
                "criticality_score": s.criticality_score,
                "annual_revenue": s.annual_revenue,
                "employee_count": s.employee_count,
                "financial_health_score": s.financial_health_score
            } for s in suppliers
        ]
        
        prompt = f"""
        You are an expert supply chain risk analyst. Analyze these suppliers and generate comprehensive risk assessment.
        
        Context: {context}
        
        Suppliers Data:
        {json.dumps(supplier_data, indent=2)}
        
        Generate detailed risk analysis with this JSON structure:
        {{
            "overall_risk_score": 0-100,
            "risk_events": [
                {{
                    "event_id": "RISK_001",
                    "supplier_id": "supplier_id",
                    "event_type": "geopolitical|financial|environmental|operational|regulatory|cyber",
                    "severity": "low|medium|high|critical",
                    "probability": 0.0-1.0,
                    "impact_score": 0.0-10.0,
                    "confidence_level": 0.0-1.0,
                    "description": "detailed_description",
                    "evidence": ["evidence1", "evidence2"],
                    "geographic_scope": "scope",
                    "predicted_timeline": "timeline",
                    "mitigation_actions": ["action1", "action2"]
                }}
            ],
            "ml_predictions": [
                {{
                    "prediction_id": "PRED_001",
                    "supplier_id": "supplier_id",
                    "prediction_type": "risk_level",
                    "prediction": "low|medium|high|critical",
                    "confidence": 0.0-1.0,
                    "feature_importance": {{
                        "financial_health": 0.3,
                        "geographic_risk": 0.25,
                        "industry_volatility": 0.2,
                        "tier_level": 0.15,
                        "company_size": 0.1
                    }}
                }}
            ],
            "recommendations": [
                "Strategic recommendation 1",
                "Strategic recommendation 2"
            ],
            "insights": {{
                "high_risk_suppliers": ["supplier_ids"],
                "emerging_risks": ["risk_descriptions"],
                "geographic_hotspots": ["regions"],
                "industry_concerns": ["industries"]
            }}
        }}
        
        Analysis Guidelines:
        - Consider geopolitical tensions, economic indicators, industry trends
        - Factor in supplier tier, criticality, and financial health
        - Generate realistic risk scenarios with evidence
        - Provide actionable mitigation strategies
        - Assign confidence levels based on data quality
        """
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                return self._fallback_analysis(suppliers)
        else:
            return self._fallback_analysis(suppliers)
    
    def _fallback_analysis(self, suppliers: List[Supplier]) -> Dict[str, Any]:
        """Fallback analysis using rule-based logic."""
        
        risk_events = []
        ml_predictions = []
        high_risk_suppliers = []
        
        for supplier in suppliers:
            # Generate risk events based on rules
            risk_score = self._calculate_risk_score(supplier)
            
            if risk_score > 70:
                high_risk_suppliers.append(supplier.id)
                
                # Generate high-risk event
                risk_events.append({
                    "event_id": f"RISK_{supplier.id}_{datetime.now().strftime('%Y%m%d')}",
                    "supplier_id": supplier.id,
                    "event_type": self._determine_risk_type(supplier),
                    "severity": "high" if risk_score > 80 else "medium",
                    "probability": min(risk_score / 100, 0.9),
                    "impact_score": min(supplier.criticality_score / 10, 10.0),
                    "confidence_level": 0.75,
                    "description": f"Elevated risk detected for {supplier.name} based on criticality and regional factors",
                    "evidence": [f"High criticality score: {supplier.criticality_score}", f"Operating in {supplier.region}"],
                    "geographic_scope": supplier.region,
                    "predicted_timeline": "2-4 weeks",
                    "mitigation_actions": [
                        "Monitor supplier closely",
                        "Identify backup suppliers",
                        "Review contract terms"
                    ]
                })
            
            # Generate ML prediction
            ml_predictions.append({
                "prediction_id": f"PRED_{supplier.id}_{datetime.now().strftime('%Y%m%d')}",
                "supplier_id": supplier.id,
                "prediction_type": "risk_level",
                "prediction": self._risk_level_from_score(risk_score),
                "confidence": 0.8,
                "feature_importance": {
                    "criticality_score": 0.4,
                    "geographic_risk": 0.25,
                    "industry_risk": 0.2,
                    "tier_level": 0.15
                }
            })
        
        overall_risk = sum(self._calculate_risk_score(s) for s in suppliers) / len(suppliers) if suppliers else 0
        
        return {
            "overall_risk_score": overall_risk,
            "risk_events": risk_events,
            "ml_predictions": ml_predictions,
            "recommendations": [
                "Diversify supplier base across regions",
                "Implement continuous monitoring for high-risk suppliers",
                "Develop contingency plans for critical suppliers",
                "Regular financial health assessments"
            ],
            "insights": {
                "high_risk_suppliers": high_risk_suppliers,
                "emerging_risks": ["Geopolitical tensions", "Supply chain disruptions"],
                "geographic_hotspots": list(set(s.region for s in suppliers if self._calculate_risk_score(s) > 60)),
                "industry_concerns": list(set(s.industry for s in suppliers if self._calculate_risk_score(s) > 60))
            }
        }
    
    def _calculate_risk_score(self, supplier: Supplier) -> float:
        """Calculate risk score based on supplier attributes."""
        base_score = supplier.criticality_score
        
        # Regional risk adjustments
        regional_risk = {
            "Asia Pacific": 15,
            "Middle East": 20,
            "South America": 10,
            "Africa": 25,
            "Europe": 5,
            "North America": 0
        }
        
        # Industry risk adjustments
        industry_risk = {
            "Mining": 20,
            "Electronics": 15,
            "Manufacturing": 10,
            "Logistics": 5,
            "Technology": 5
        }
        
        # Tier risk adjustments
        tier_risk = {
            "tier_1": 10,
            "tier_2": 5,
            "tier_3": 0
        }
        
        risk_adjustment = (
            regional_risk.get(supplier.region, 10) +
            industry_risk.get(supplier.industry, 10) +
            tier_risk.get(supplier.tier, 5)
        )
        
        # Financial health adjustment
        if supplier.financial_health_score:
            if supplier.financial_health_score < 50:
                risk_adjustment += 20
            elif supplier.financial_health_score < 70:
                risk_adjustment += 10
        
        return min(base_score + risk_adjustment, 100)
    
    def _determine_risk_type(self, supplier: Supplier) -> str:
        """Determine primary risk type based on supplier characteristics."""
        if supplier.region in ["Middle East", "Asia Pacific"]:
            return "geopolitical"
        elif supplier.industry in ["Mining", "Electronics"]:
            return "operational"
        elif supplier.financial_health_score and supplier.financial_health_score < 60:
            return "financial"
        else:
            return "operational"
    
    def _risk_level_from_score(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"
    
    def generate_insights(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate additional insights from analysis results."""
        
        if not self.client:
            return {"insights": "AI insights unavailable - using rule-based analysis"}
        
        prompt = f"""
        Based on this supply chain risk analysis, provide strategic insights and recommendations:
        
        Analysis Results:
        {json.dumps(analysis_result, indent=2)[:3000]}
        
        Provide insights in JSON format:
        {{
            "executive_summary": "Brief summary for executives",
            "key_risks": ["Top 3 risks"],
            "strategic_actions": ["Immediate actions needed"],
            "monitoring_priorities": ["What to monitor closely"],
            "long_term_strategy": "Long-term supply chain strategy recommendations"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"Failed to generate insights: {str(e)}"}