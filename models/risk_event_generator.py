"""
Risk event generation system for automatic creation of risk events from ML predictions.
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .supplier import Supplier
from .risk_event import RiskEvent, RiskLevel, DataSource
from .ml_prediction import MLPrediction
from .scraped_data import ScrapedData

logger = logging.getLogger(__name__)


class RiskEventGenerator:
    """
    Generates risk events automatically from ML predictions and collected data.
    """
    
    def __init__(self):
        """Initialize the risk event generator."""
        self.event_type_mapping = self._get_event_type_mapping()
        self.severity_thresholds = self._get_severity_thresholds()
        self.evidence_extractors = self._get_evidence_extractors()
        
    def _get_event_type_mapping(self) -> Dict[str, str]:
        """Map risk indicators to event types."""
        return {
            # Geopolitical indicators
            'political': 'geopolitical',
            'sanctions': 'geopolitical',
            'trade_war': 'geopolitical',
            'conflict': 'geopolitical',
            'instability': 'geopolitical',
            'government': 'geopolitical',
            
            # Financial indicators
            'bankruptcy': 'financial',
            'debt': 'financial',
            'revenue': 'financial',
            'profit': 'financial',
            'cash_flow': 'financial',
            'credit': 'financial',
            'financial_distress': 'financial',
            
            # Environmental indicators
            'climate': 'environmental',
            'weather': 'environmental',
            'natural_disaster': 'environmental',
            'flood': 'environmental',
            'earthquake': 'environmental',
            'hurricane': 'environmental',
            'drought': 'environmental',
            
            # Operational indicators
            'production': 'operational',
            'capacity': 'operational',
            'quality': 'operational',
            'delivery': 'operational',
            'logistics': 'operational',
            'supply': 'operational',
            'manufacturing': 'operational',
            
            # Regulatory indicators
            'regulation': 'regulatory',
            'compliance': 'regulatory',
            'legal': 'regulatory',
            'lawsuit': 'regulatory',
            'investigation': 'regulatory',
            'fine': 'regulatory',
            
            # Cyber indicators
            'cyber': 'cyber',
            'hack': 'cyber',
            'breach': 'cyber',
            'security': 'cyber',
            'ransomware': 'cyber',
            'data_theft': 'cyber'
        }
    
    def _get_severity_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Define thresholds for severity level assignment."""
        return {
            'probability': {
                'critical': 0.8,
                'high': 0.6,
                'medium': 0.4
            },
            'impact_score': {
                'critical': 8.0,
                'high': 6.0,
                'medium': 4.0
            },
            'confidence': {
                'critical': 0.8,
                'high': 0.6,
                'medium': 0.4
            }
        }
    
    def _get_evidence_extractors(self) -> Dict[str, callable]:
        """Define evidence extraction functions for different data sources."""
        return {
            'ml_prediction': self._extract_ml_evidence,
            'scraped_data': self._extract_scraped_evidence,
            'supplier_data': self._extract_supplier_evidence
        }
    
    def generate_risk_events_from_predictions(
        self, 
        predictions: List[MLPrediction],
        suppliers: List[Supplier],
        scraped_data: List[ScrapedData] = None
    ) -> List[RiskEvent]:
        """
        Generate risk events from ML predictions.
        
        Args:
            predictions: List of ML predictions
            suppliers: List of suppliers
            scraped_data: Optional scraped data for evidence
            
        Returns:
            List of generated risk events
        """
        risk_events = []
        supplier_map = {s.id: s for s in suppliers}
        scraped_data_map = self._group_scraped_data_by_supplier(scraped_data or [])
        
        for prediction in predictions:
            if prediction.supplier_id not in supplier_map:
                logger.warning(f"Supplier {prediction.supplier_id} not found, skipping prediction")
                continue
            
            supplier = supplier_map[prediction.supplier_id]
            supplier_scraped_data = scraped_data_map.get(prediction.supplier_id, [])
            
            # Generate risk events based on prediction
            events = self._create_events_from_prediction(
                prediction, supplier, supplier_scraped_data
            )
            risk_events.extend(events)
        
        logger.info(f"Generated {len(risk_events)} risk events from {len(predictions)} predictions")
        return risk_events
    
    def _group_scraped_data_by_supplier(self, scraped_data: List[ScrapedData]) -> Dict[str, List[ScrapedData]]:
        """Group scraped data by supplier ID."""
        grouped = {}
        for data in scraped_data:
            if data.supplier_id not in grouped:
                grouped[data.supplier_id] = []
            grouped[data.supplier_id].append(data)
        return grouped
    
    def _create_events_from_prediction(
        self,
        prediction: MLPrediction,
        supplier: Supplier,
        scraped_data: List[ScrapedData]
    ) -> List[RiskEvent]:
        """
        Create risk events from a single ML prediction.
        
        Args:
            prediction: ML prediction
            supplier: Supplier object
            scraped_data: Scraped data for the supplier
            
        Returns:
            List of risk events
        """
        events = []
        
        # Determine if prediction warrants risk event creation
        if not self._should_create_risk_event(prediction):
            return events
        
        # Determine event types based on prediction and data
        event_types = self._determine_event_types(prediction, supplier, scraped_data)
        
        for event_type in event_types:
            event = self._create_single_risk_event(
                prediction, supplier, scraped_data, event_type
            )
            if event:
                events.append(event)
        
        return events
    
    def _should_create_risk_event(self, prediction: MLPrediction) -> bool:
        """
        Determine if a prediction should generate a risk event.
        
        Args:
            prediction: ML prediction
            
        Returns:
            True if risk event should be created
        """
        # Check if prediction indicates significant risk
        if prediction.prediction in ['high', 'critical']:
            return True
        
        # Check probability and impact from metadata
        metadata = prediction.model_metadata or {}
        probability = metadata.get('probability', 0)
        impact_score = metadata.get('impact_score', 0)
        
        # Create event if probability or impact is high
        if probability > 0.6 or impact_score > 6.0:
            return True
        
        # Check confidence level
        if prediction.confidence > 0.8 and prediction.prediction == 'medium':
            return True
        
        return False
    
    def _determine_event_types(
        self,
        prediction: MLPrediction,
        supplier: Supplier,
        scraped_data: List[ScrapedData]
    ) -> List[str]:
        """
        Determine event types based on available data.
        
        Args:
            prediction: ML prediction
            supplier: Supplier object
            scraped_data: Scraped data for the supplier
            
        Returns:
            List of event types
        """
        event_types = set()
        
        # Analyze feature importance from prediction
        feature_importance = prediction.feature_importance or {}
        
        # Map high-importance features to event types
        for feature, importance in feature_importance.items():
            if importance > 0.3:  # High importance threshold
                if 'country' in feature.lower() or 'political' in feature.lower():
                    event_types.add('geopolitical')
                elif 'financial' in feature.lower() or 'revenue' in feature.lower():
                    event_types.add('financial')
                elif 'industry' in feature.lower():
                    # Map industry to likely event type
                    industry_event_map = {
                        'Oil & Gas': 'environmental',
                        'Mining': 'environmental',
                        'Manufacturing': 'operational',
                        'Software': 'cyber',
                        'Defense': 'geopolitical'
                    }
                    event_types.add(industry_event_map.get(supplier.industry, 'operational'))
        
        # Analyze scraped data for event type indicators
        for data in scraped_data:
            risk_indicators = data.risk_indicators or []
            for indicator in risk_indicators:
                indicator_lower = indicator.lower()
                for keyword, event_type in self.event_type_mapping.items():
                    if keyword in indicator_lower:
                        event_types.add(event_type)
        
        # Default to operational if no specific type identified
        if not event_types:
            event_types.add('operational')
        
        return list(event_types)
    
    def _create_single_risk_event(
        self,
        prediction: MLPrediction,
        supplier: Supplier,
        scraped_data: List[ScrapedData],
        event_type: str
    ) -> Optional[RiskEvent]:
        """
        Create a single risk event.
        
        Args:
            prediction: ML prediction
            supplier: Supplier object
            scraped_data: Scraped data for the supplier
            event_type: Type of risk event
            
        Returns:
            RiskEvent object or None if creation fails
        """
        try:
            # Generate unique event ID
            event_id = f"RISK_{supplier.id}_{event_type.upper()}_{uuid.uuid4().hex[:8]}"
            
            # Determine severity level
            severity = self._calculate_severity(prediction, event_type)
            
            # Extract probability and impact from prediction
            metadata = prediction.model_metadata or {}
            probability = metadata.get('probability', prediction.confidence)
            impact_score = metadata.get('impact_score', 5.0)
            
            # Generate description
            description = self._generate_event_description(
                prediction, supplier, event_type, severity
            )
            
            # Collect evidence
            evidence = self._collect_evidence(prediction, supplier, scraped_data, event_type)
            
            # Determine data sources
            data_sources = self._determine_data_sources(prediction, scraped_data)
            
            # Calculate geographic scope and timeline
            geographic_scope = self._determine_geographic_scope(supplier, event_type)
            predicted_timeline = self._predict_timeline(severity, event_type)
            
            # Generate mitigation actions
            mitigation_actions = self._generate_mitigation_actions(
                event_type, severity, supplier
            )
            
            # Create risk event
            risk_event = RiskEvent(
                event_id=event_id,
                supplier_id=supplier.id,
                event_type=event_type,
                severity=severity,
                probability=min(1.0, probability),
                impact_score=min(10.0, impact_score),
                confidence_level=prediction.confidence,
                description=description,
                evidence=evidence,
                data_sources=data_sources,
                geographic_scope=geographic_scope,
                predicted_timeline=predicted_timeline,
                mitigation_actions=mitigation_actions,
                status="active"
            )
            
            return risk_event
            
        except Exception as e:
            logger.error(f"Failed to create risk event for {supplier.id}: {e}")
            return None
    
    def _calculate_severity(self, prediction: MLPrediction, event_type: str) -> RiskLevel:
        """
        Calculate severity level based on prediction and event type.
        
        Args:
            prediction: ML prediction
            event_type: Type of risk event
            
        Returns:
            RiskLevel enum value
        """
        metadata = prediction.model_metadata or {}
        probability = metadata.get('probability', prediction.confidence)
        impact_score = metadata.get('impact_score', 5.0)
        
        # Get thresholds
        prob_thresholds = self.severity_thresholds['probability']
        impact_thresholds = self.severity_thresholds['impact_score']
        conf_thresholds = self.severity_thresholds['confidence']
        
        # Calculate severity based on multiple factors
        severity_score = 0
        
        # Probability contribution
        if probability >= prob_thresholds['critical']:
            severity_score += 3
        elif probability >= prob_thresholds['high']:
            severity_score += 2
        elif probability >= prob_thresholds['medium']:
            severity_score += 1
        
        # Impact contribution
        if impact_score >= impact_thresholds['critical']:
            severity_score += 3
        elif impact_score >= impact_thresholds['high']:
            severity_score += 2
        elif impact_score >= impact_thresholds['medium']:
            severity_score += 1
        
        # Confidence contribution
        if prediction.confidence >= conf_thresholds['critical']:
            severity_score += 1
        
        # Event type modifier
        high_impact_types = ['geopolitical', 'financial', 'cyber']
        if event_type in high_impact_types:
            severity_score += 1
        
        # Map score to severity level
        if severity_score >= 6:
            return RiskLevel.CRITICAL
        elif severity_score >= 4:
            return RiskLevel.HIGH
        elif severity_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_event_description(
        self,
        prediction: MLPrediction,
        supplier: Supplier,
        event_type: str,
        severity: RiskLevel
    ) -> str:
        """
        Generate a descriptive text for the risk event.
        
        Args:
            prediction: ML prediction
            supplier: Supplier object
            event_type: Type of risk event
            severity: Severity level
            
        Returns:
            Event description string
        """
        # Base description templates
        templates = {
            'geopolitical': f"Geopolitical risk detected for supplier {supplier.name} in {supplier.country}. Political instability or regulatory changes may impact operations.",
            'financial': f"Financial risk identified for supplier {supplier.name}. Company financial health indicators suggest potential disruption to supply capabilities.",
            'environmental': f"Environmental risk detected affecting supplier {supplier.name} in {supplier.region}. Weather or climate-related factors may disrupt operations.",
            'operational': f"Operational risk identified for supplier {supplier.name}. Production, quality, or logistics issues may impact supply chain performance.",
            'regulatory': f"Regulatory compliance risk detected for supplier {supplier.name}. Legal or regulatory changes may affect business operations.",
            'cyber': f"Cybersecurity risk identified for supplier {supplier.name}. Security vulnerabilities or incidents may disrupt digital operations."
        }
        
        base_description = templates.get(event_type, f"Risk event detected for supplier {supplier.name}")
        
        # Add severity context
        severity_context = {
            RiskLevel.CRITICAL: "Immediate attention required - high probability of significant business impact.",
            RiskLevel.HIGH: "Elevated risk level - proactive monitoring and mitigation recommended.",
            RiskLevel.MEDIUM: "Moderate risk level - continued monitoring advised.",
            RiskLevel.LOW: "Low risk level - routine monitoring sufficient."
        }
        
        description = f"{base_description} {severity_context.get(severity, '')}"
        
        # Add prediction confidence context
        if prediction.confidence > 0.8:
            description += f" High confidence prediction (confidence: {prediction.confidence:.2f})."
        elif prediction.confidence > 0.6:
            description += f" Moderate confidence prediction (confidence: {prediction.confidence:.2f})."
        
        return description
    
    def _collect_evidence(
        self,
        prediction: MLPrediction,
        supplier: Supplier,
        scraped_data: List[ScrapedData],
        event_type: str
    ) -> List[str]:
        """
        Collect evidence supporting the risk event.
        
        Args:
            prediction: ML prediction
            supplier: Supplier object
            scraped_data: Scraped data for the supplier
            event_type: Type of risk event
            
        Returns:
            List of evidence strings
        """
        evidence = []
        
        # ML prediction evidence
        ml_evidence = self._extract_ml_evidence(prediction)
        evidence.extend(ml_evidence)
        
        # Scraped data evidence
        scraped_evidence = self._extract_scraped_evidence(scraped_data, event_type)
        evidence.extend(scraped_evidence)
        
        # Supplier data evidence
        supplier_evidence = self._extract_supplier_evidence(supplier, event_type)
        evidence.extend(supplier_evidence)
        
        # Remove duplicates and empty entries
        evidence = list(set([e for e in evidence if e.strip()]))
        
        return evidence[:10]  # Limit to top 10 pieces of evidence
    
    def _extract_ml_evidence(self, prediction: MLPrediction) -> List[str]:
        """Extract evidence from ML prediction."""
        evidence = []
        
        # Feature importance evidence
        feature_importance = prediction.feature_importance or {}
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for feature, importance in top_features:
            evidence.append(f"High importance factor: {feature} (importance: {importance:.2f})")
        
        # Prediction confidence
        evidence.append(f"ML model prediction: {prediction.prediction} (confidence: {prediction.confidence:.2f})")
        
        # Model metadata
        metadata = prediction.model_metadata or {}
        if 'composite_risk_score' in metadata:
            evidence.append(f"Composite risk score: {metadata['composite_risk_score']:.2f}")
        
        return evidence
    
    def _extract_scraped_evidence(self, scraped_data: List[ScrapedData], event_type: str) -> List[str]:
        """Extract evidence from scraped data."""
        evidence = []
        
        for data in scraped_data:
            # Risk indicators
            risk_indicators = data.risk_indicators or []
            relevant_indicators = [
                indicator for indicator in risk_indicators
                if any(keyword in indicator.lower() for keyword in self.event_type_mapping.keys())
            ]
            
            for indicator in relevant_indicators[:3]:  # Top 3 per source
                evidence.append(f"Risk indicator from {data.source_url}: {indicator}")
            
            # Sentiment analysis
            if data.sentiment_score and data.sentiment_score < -0.3:
                evidence.append(f"Negative sentiment detected in content from {data.source_url} (score: {data.sentiment_score:.2f})")
            
            # Keywords
            keywords = data.keywords or []
            event_keywords = [
                kw for kw in keywords
                if any(event_word in kw.lower() for event_word in self.event_type_mapping.keys())
            ]
            
            if event_keywords:
                evidence.append(f"Relevant keywords found: {', '.join(event_keywords[:3])}")
        
        return evidence
    
    def _extract_supplier_evidence(self, supplier: Supplier, event_type: str) -> List[str]:
        """Extract evidence from supplier data."""
        evidence = []
        
        # Financial health
        if supplier.financial_health_score and supplier.financial_health_score < 40:
            evidence.append(f"Low financial health score: {supplier.financial_health_score}/100")
        
        # Geographic risk
        high_risk_countries = ['Russia', 'Iran', 'North Korea', 'Venezuela']
        if supplier.country in high_risk_countries:
            evidence.append(f"Supplier located in high-risk country: {supplier.country}")
        
        # Industry risk
        high_risk_industries = ['Oil & Gas', 'Mining', 'Defense']
        if supplier.industry in high_risk_industries:
            evidence.append(f"Supplier operates in high-risk industry: {supplier.industry}")
        
        # Size risk
        if supplier.employee_count and supplier.employee_count < 50:
            evidence.append(f"Small supplier size may indicate higher risk (employees: {supplier.employee_count})")
        
        # Criticality
        if supplier.criticality_score > 80:
            evidence.append(f"High criticality supplier (score: {supplier.criticality_score}/100)")
        
        return evidence
    
    def _determine_data_sources(
        self,
        prediction: MLPrediction,
        scraped_data: List[ScrapedData]
    ) -> List[DataSource]:
        """Determine data sources used for the risk event."""
        sources = [DataSource.ML_PREDICTION]
        
        if scraped_data:
            sources.append(DataSource.WEB_SCRAPING)
        
        # Could add API_DATA if we had API data in the prediction
        # Could add FILE_UPLOAD if we had file-based data
        
        return sources
    
    def _determine_geographic_scope(self, supplier: Supplier, event_type: str) -> str:
        """Determine geographic scope of the risk event."""
        if event_type == 'geopolitical':
            return f"{supplier.country}, {supplier.region}"
        elif event_type == 'environmental':
            return f"{supplier.region}"
        else:
            return supplier.country
    
    def _predict_timeline(self, severity: RiskLevel, event_type: str) -> str:
        """Predict timeline for risk materialization."""
        timeline_map = {
            RiskLevel.CRITICAL: {
                'geopolitical': '1-2 weeks',
                'financial': '2-4 weeks',
                'environmental': '1-3 days',
                'operational': '1-2 weeks',
                'regulatory': '4-8 weeks',
                'cyber': '1-7 days'
            },
            RiskLevel.HIGH: {
                'geopolitical': '2-6 weeks',
                'financial': '1-3 months',
                'environmental': '1-2 weeks',
                'operational': '2-6 weeks',
                'regulatory': '2-6 months',
                'cyber': '1-4 weeks'
            },
            RiskLevel.MEDIUM: {
                'geopolitical': '1-6 months',
                'financial': '3-12 months',
                'environmental': '1-3 months',
                'operational': '1-6 months',
                'regulatory': '6-18 months',
                'cyber': '1-3 months'
            },
            RiskLevel.LOW: {
                'geopolitical': '6-18 months',
                'financial': '12+ months',
                'environmental': '3-12 months',
                'operational': '6-18 months',
                'regulatory': '12+ months',
                'cyber': '3-12 months'
            }
        }
        
        return timeline_map.get(severity, {}).get(event_type, '3-6 months')
    
    def _generate_mitigation_actions(
        self,
        event_type: str,
        severity: RiskLevel,
        supplier: Supplier
    ) -> List[str]:
        """Generate recommended mitigation actions."""
        actions = []
        
        # Base actions by event type
        action_templates = {
            'geopolitical': [
                'Monitor political developments and policy changes',
                'Identify alternative suppliers in stable regions',
                'Assess contract terms for force majeure clauses',
                'Engage with local government relations teams'
            ],
            'financial': [
                'Review supplier financial statements and credit ratings',
                'Implement enhanced payment terms monitoring',
                'Identify backup suppliers for critical components',
                'Consider supplier financing or support programs'
            ],
            'environmental': [
                'Monitor weather forecasts and climate risks',
                'Assess supplier facility locations for environmental exposure',
                'Develop contingency plans for weather-related disruptions',
                'Consider climate-resilient supplier alternatives'
            ],
            'operational': [
                'Increase quality monitoring and inspections',
                'Review production capacity and delivery schedules',
                'Implement enhanced communication protocols',
                'Assess need for inventory buffer increases'
            ],
            'regulatory': [
                'Monitor regulatory changes and compliance requirements',
                'Review supplier certifications and compliance status',
                'Assess impact of new regulations on supply chain',
                'Engage legal counsel for regulatory guidance'
            ],
            'cyber': [
                'Assess supplier cybersecurity posture and controls',
                'Implement enhanced data security requirements',
                'Review incident response and business continuity plans',
                'Consider cyber insurance coverage evaluation'
            ]
        }
        
        base_actions = action_templates.get(event_type, [
            'Increase monitoring and communication with supplier',
            'Review contract terms and risk allocation',
            'Assess alternative supplier options',
            'Implement enhanced risk monitoring procedures'
        ])
        
        # Add severity-specific actions
        if severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            actions.extend([
                'Activate crisis management protocols',
                'Notify key stakeholders and decision makers',
                'Accelerate alternative supplier qualification'
            ])
        
        # Add supplier-specific actions
        if supplier.criticality_score > 80:
            actions.append('Prioritize due to high supplier criticality')
        
        if supplier.tier == 'tier_1':
            actions.append('Coordinate with tier-2 suppliers for visibility')
        
        # Combine and limit actions
        all_actions = base_actions + actions
        return all_actions[:8]  # Limit to top 8 actions
    
    def categorize_existing_events(self, risk_events: List[RiskEvent]) -> Dict[str, List[RiskEvent]]:
        """
        Categorize existing risk events by type.
        
        Args:
            risk_events: List of risk events to categorize
            
        Returns:
            Dictionary mapping event types to lists of events
        """
        categorized = {}
        
        for event in risk_events:
            event_type = event.event_type
            if event_type not in categorized:
                categorized[event_type] = []
            categorized[event_type].append(event)
        
        return categorized
    
    def get_generation_statistics(self, risk_events: List[RiskEvent]) -> Dict[str, Any]:
        """
        Get statistics about generated risk events.
        
        Args:
            risk_events: List of generated risk events
            
        Returns:
            Dictionary with generation statistics
        """
        if not risk_events:
            return {
                'total_events': 0,
                'events_by_type': {},
                'events_by_severity': {},
                'average_confidence': 0,
                'average_probability': 0,
                'average_impact': 0
            }
        
        # Count by type
        events_by_type = {}
        for event in risk_events:
            event_type = event.event_type
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
        
        # Count by severity
        events_by_severity = {}
        for event in risk_events:
            severity = event.severity
            events_by_severity[severity] = events_by_severity.get(severity, 0) + 1
        
        # Calculate averages
        avg_confidence = sum(event.confidence_level for event in risk_events) / len(risk_events)
        avg_probability = sum(event.probability for event in risk_events) / len(risk_events)
        avg_impact = sum(event.impact_score for event in risk_events) / len(risk_events)
        
        return {
            'total_events': len(risk_events),
            'events_by_type': events_by_type,
            'events_by_severity': events_by_severity,
            'average_confidence': round(avg_confidence, 3),
            'average_probability': round(avg_probability, 3),
            'average_impact': round(avg_impact, 2),
            'high_priority_events': len([e for e in risk_events if e.is_high_priority()]),
            'generation_timestamp': datetime.now().isoformat()
        }