"""
Simple ML prediction service for supply chain risk assessment.
Provides basic risk scoring and predictions that can be called by agents as needed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from models.supplier import Supplier
from models.scraped_data import ScrapedData
from models.risk_event import RiskEvent, RiskLevel
from models.ml_prediction import MLPrediction

logger = logging.getLogger(__name__)


class SimpleMLPredictor:
    """
    Simple ML prediction service for supply chain risk assessment.
    Provides basic risk scoring that agents can call when needed.
    """
    
    def __init__(self):
        """Initialize the simple ML predictor."""
        self.scaler = StandardScaler()
        self.risk_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.impact_regressor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model_version = "1.0.0"
        self.is_trained = False
        
        # Risk scoring weights
        self.country_risk_scores = self._get_country_risk_scores()
        self.industry_risk_scores = self._get_industry_risk_scores()
    
    def _get_country_risk_scores(self) -> Dict[str, float]:
        """Get basic country risk scores."""
        return {
            'Germany': 0.2, 'United States': 0.2, 'Canada': 0.2, 'Japan': 0.2,
            'China': 0.5, 'India': 0.5, 'Brazil': 0.5, 'Mexico': 0.5,
            'Russia': 0.8, 'Iran': 0.9, 'North Korea': 0.95
        }
    
    def _get_industry_risk_scores(self) -> Dict[str, float]:
        """Get basic industry risk scores."""
        return {
            'Software': 0.1, 'Healthcare': 0.2, 'Manufacturing': 0.4,
            'Oil & Gas': 0.7, 'Mining': 0.8, 'Defense': 0.9
        }
    
    def calculate_risk_score(self, supplier: Supplier, 
                           scraped_data: List[ScrapedData] = None) -> Dict[str, Any]:
        """
        Calculate basic risk score for a supplier using rule-based approach.
        
        Args:
            supplier: Supplier object
            scraped_data: Optional scraped data for the supplier
            
        Returns:
            Dictionary with risk assessment results
        """
        # Base risk factors
        country_risk = self.country_risk_scores.get(supplier.country, 0.5)
        industry_risk = self.industry_risk_scores.get(supplier.industry, 0.5)
        
        # Financial health factor
        financial_risk = 1.0 - (supplier.financial_health_score or 50) / 100
        
        # Size factor (smaller companies = higher risk)
        if supplier.employee_count:
            if supplier.employee_count < 100:
                size_risk = 0.8
            elif supplier.employee_count < 1000:
                size_risk = 0.5
            else:
                size_risk = 0.2
        else:
            size_risk = 0.6
        
        # Content-based risk (if scraped data available)
        content_risk = 0.5
        if scraped_data:
            negative_sentiment_count = sum(1 for data in scraped_data 
                                         if data.sentiment_score and data.sentiment_score < -0.3)
            risk_indicator_count = sum(len(data.risk_indicators) for data in scraped_data)
            
            if scraped_data:
                content_risk = min(1.0, (negative_sentiment_count + risk_indicator_count * 0.1) / len(scraped_data))
        
        # Calculate composite risk score using configurable weights
        weights = getattr(self, 'risk_weights', {
            'country': 0.25,
            'industry': 0.20,
            'financial': 0.25,
            'size': 0.15,
            'content': 0.15
        })
        
        composite_risk = (
            country_risk * weights['country'] +
            industry_risk * weights['industry'] +
            financial_risk * weights['financial'] +
            size_risk * weights['size'] +
            content_risk * weights['content']
        )
        
        # Determine risk level using configurable thresholds
        thresholds = getattr(self, 'risk_thresholds', {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4
        })
        
        if composite_risk >= thresholds['critical']:
            risk_level = 'critical'
        elif composite_risk >= thresholds['high']:
            risk_level = 'high'
        elif composite_risk >= thresholds['medium']:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Calculate impact score (0-10 scale)
        impact_score = min(10.0, composite_risk * 10 * (supplier.criticality_score / 100))
        
        # Calculate probability
        probability = min(1.0, composite_risk * 1.2)
        
        return {
            'supplier_id': supplier.id,
            'risk_level': risk_level,
            'composite_risk_score': composite_risk,
            'impact_score': impact_score,
            'probability': probability,
            'confidence': 0.7,  # Fixed confidence for rule-based approach
            'risk_factors': {
                'country_risk': country_risk,
                'industry_risk': industry_risk,
                'financial_risk': financial_risk,
                'size_risk': size_risk,
                'content_risk': content_risk
            }
        }
    
    def predict_supplier_risk(self, supplier: Supplier, 
                            scraped_data: List[ScrapedData] = None) -> MLPrediction:
        """
        Create ML prediction object for a supplier.
        
        Args:
            supplier: Supplier object
            scraped_data: Optional scraped data for the supplier
            
        Returns:
            MLPrediction object with results
        """
        risk_assessment = self.calculate_risk_score(supplier, scraped_data)
        
        prediction = MLPrediction(
            prediction_id=f"PRED_{supplier.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            supplier_id=supplier.id,
            model_version=self.model_version,
            prediction_type="risk_assessment",
            prediction=risk_assessment['risk_level'],
            confidence=risk_assessment['confidence'],
            probability_scores={
                'low': 0.1 if risk_assessment['risk_level'] != 'low' else 0.7,
                'medium': 0.1 if risk_assessment['risk_level'] != 'medium' else 0.7,
                'high': 0.1 if risk_assessment['risk_level'] != 'high' else 0.7,
                'critical': 0.1 if risk_assessment['risk_level'] != 'critical' else 0.7
            },
            feature_importance={
                'country_risk': risk_assessment['risk_factors']['country_risk'],
                'industry_risk': risk_assessment['risk_factors']['industry_risk'],
                'financial_risk': risk_assessment['risk_factors']['financial_risk'],
                'size_risk': risk_assessment['risk_factors']['size_risk'],
                'content_risk': risk_assessment['risk_factors']['content_risk']
            },
            input_features={
                'country': supplier.country,
                'industry': supplier.industry,
                'financial_health_score': supplier.financial_health_score,
                'employee_count': supplier.employee_count,
                'criticality_score': supplier.criticality_score
            },
            model_metadata={
                'approach': 'rule_based',
                'version': self.model_version,
                'composite_risk_score': risk_assessment['composite_risk_score'],
                'impact_score': risk_assessment['impact_score'],
                'probability': risk_assessment['probability']
            }
        )
        
        return prediction
    
    def batch_predict(self, suppliers: List[Supplier], 
                     scraped_data_map: Dict[str, List[ScrapedData]] = None) -> List[MLPrediction]:
        """
        Predict risk for multiple suppliers.
        
        Args:
            suppliers: List of Supplier objects
            scraped_data_map: Optional mapping of supplier_id to scraped data
            
        Returns:
            List of MLPrediction objects
        """
        predictions = []
        scraped_data_map = scraped_data_map or {}
        
        for supplier in suppliers:
            supplier_scraped_data = scraped_data_map.get(supplier.id, [])
            prediction = self.predict_supplier_risk(supplier, supplier_scraped_data)
            predictions.append(prediction)
        
        logger.info(f"Generated predictions for {len(suppliers)} suppliers")
        return predictions
    
    def get_risk_summary(self, suppliers: List[Supplier], 
                        scraped_data_map: Dict[str, List[ScrapedData]] = None) -> Dict[str, Any]:
        """
        Get risk summary for a list of suppliers.
        
        Args:
            suppliers: List of Supplier objects
            scraped_data_map: Optional mapping of supplier_id to scraped data
            
        Returns:
            Dictionary with risk summary statistics
        """
        predictions = self.batch_predict(suppliers, scraped_data_map)
        
        risk_levels = [pred.prediction for pred in predictions]
        risk_counts = {level: risk_levels.count(level) for level in ['low', 'medium', 'high', 'critical']}
        
        impact_scores = [pred.model_metadata.get('impact_score', 0) for pred in predictions]
        probabilities = [pred.model_metadata.get('probability', 0) for pred in predictions]
        
        high_risk_suppliers = [
            pred.supplier_id for pred in predictions 
            if pred.prediction in ['high', 'critical']
        ]
        
        return {
            'total_suppliers': len(suppliers),
            'risk_distribution': risk_counts,
            'high_risk_count': len(high_risk_suppliers),
            'high_risk_suppliers': high_risk_suppliers,
            'average_impact_score': np.mean(impact_scores) if impact_scores else 0,
            'average_probability': np.mean(probabilities) if probabilities else 0,
            'max_impact_score': max(impact_scores) if impact_scores else 0,
            'predictions_generated': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
    
    def train_simple_models(self, suppliers: List[Supplier], 
                          risk_events: List[RiskEvent]) -> Dict[str, Any]:
        """
        Train simple models if historical data is available.
        This is optional - the rule-based approach works without training.
        
        Args:
            suppliers: List of Supplier objects
            risk_events: List of RiskEvent objects for training
            
        Returns:
            Training summary
        """
        if not suppliers or not risk_events:
            logger.info("No training data available, using rule-based approach only")
            return {'status': 'rule_based_only', 'trained': False}
        
        # Create simple features for training
        features = []
        targets = []
        
        for supplier in suppliers:
            supplier_events = [e for e in risk_events if e.supplier_id == supplier.id]
            if not supplier_events:
                continue
                
            # Simple feature vector
            feature_vector = [
                self.country_risk_scores.get(supplier.country, 0.5),
                self.industry_risk_scores.get(supplier.industry, 0.5),
                1.0 - (supplier.financial_health_score or 50) / 100,
                supplier.criticality_score / 100,
                1 if (supplier.employee_count or 0) < 100 else 0
            ]
            
            # Target: highest risk level
            risk_levels = [e.severity for e in supplier_events]
            risk_priority = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
            target = max(risk_levels, key=lambda x: risk_priority.get(x, 0))
            
            features.append(feature_vector)
            targets.append(target)
        
        if len(features) < 5:  # Need minimum samples
            logger.info("Insufficient training data, using rule-based approach only")
            return {'status': 'insufficient_data', 'trained': False}
        
        # Train simple classifier
        X = np.array(features)
        y = targets
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.risk_classifier.fit(X_train_scaled, y_train)
        
        # Simple evaluation
        train_score = self.risk_classifier.score(X_train_scaled, y_train)
        test_score = self.risk_classifier.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        logger.info(f"Simple model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        
        return {
            'status': 'trained',
            'trained': True,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_samples': len(features),
            'training_date': datetime.now().isoformat()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'approach': 'rule_based_with_optional_ml',
            'supported_predictions': ['risk_level', 'impact_score', 'probability'],
            'country_risk_coverage': len(self.country_risk_scores),
            'industry_risk_coverage': len(self.industry_risk_scores)
        }
    
    # Agent-configurable methods for dynamic adjustments
    
    def update_country_risk_score(self, country: str, risk_score: float) -> bool:
        """
        Allow agents to update country risk scores dynamically.
        
        Args:
            country: Country name
            risk_score: Risk score between 0.0 and 1.0
            
        Returns:
            True if update was successful
        """
        if not (0.0 <= risk_score <= 1.0):
            logger.error(f"Invalid risk score {risk_score}. Must be between 0.0 and 1.0")
            return False
        
        old_score = self.country_risk_scores.get(country, "not found")
        self.country_risk_scores[country] = risk_score
        logger.info(f"Updated country risk: {country} {old_score} -> {risk_score}")
        return True
    
    def update_industry_risk_score(self, industry: str, risk_score: float) -> bool:
        """
        Allow agents to update industry risk scores dynamically.
        
        Args:
            industry: Industry name
            risk_score: Risk score between 0.0 and 1.0
            
        Returns:
            True if update was successful
        """
        if not (0.0 <= risk_score <= 1.0):
            logger.error(f"Invalid risk score {risk_score}. Must be between 0.0 and 1.0")
            return False
        
        old_score = self.industry_risk_scores.get(industry, "not found")
        self.industry_risk_scores[industry] = risk_score
        logger.info(f"Updated industry risk: {industry} {old_score} -> {risk_score}")
        return True
    
    def update_risk_weights(self, weights: Dict[str, float]) -> bool:
        """
        Allow agents to update risk calculation weights dynamically.
        
        Args:
            weights: Dictionary with keys: country, industry, financial, size, content
            
        Returns:
            True if update was successful
        """
        required_keys = {'country', 'industry', 'financial', 'size', 'content'}
        
        if not all(key in weights for key in required_keys):
            logger.error(f"Missing required weight keys. Need: {required_keys}")
            return False
        
        if not all(0.0 <= weight <= 1.0 for weight in weights.values()):
            logger.error("All weights must be between 0.0 and 1.0")
            return False
        
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.error(f"Weights must sum to 1.0, got {total_weight}")
            return False
        
        # Store weights as instance variable
        self.risk_weights = weights
        logger.info(f"Updated risk weights: {weights}")
        return True
    
    def update_risk_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """
        Allow agents to update risk level thresholds dynamically.
        
        Args:
            thresholds: Dictionary with keys: critical, high, medium (low is implicit)
            
        Returns:
            True if update was successful
        """
        required_keys = {'critical', 'high', 'medium'}
        
        if not all(key in thresholds for key in required_keys):
            logger.error(f"Missing required threshold keys. Need: {required_keys}")
            return False
        
        if not all(0.0 <= threshold <= 1.0 for threshold in thresholds.values()):
            logger.error("All thresholds must be between 0.0 and 1.0")
            return False
        
        # Validate threshold ordering
        if not (thresholds['critical'] > thresholds['high'] > thresholds['medium']):
            logger.error("Thresholds must be in order: critical > high > medium")
            return False
        
        # Store thresholds as instance variable
        self.risk_thresholds = thresholds
        logger.info(f"Updated risk thresholds: {thresholds}")
        return True
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current configuration that agents can modify.
        
        Returns:
            Dictionary with current configuration
        """
        return {
            'country_risk_scores': self.country_risk_scores.copy(),
            'industry_risk_scores': self.industry_risk_scores.copy(),
            'risk_weights': getattr(self, 'risk_weights', {
                'country': 0.25, 'industry': 0.20, 'financial': 0.25, 
                'size': 0.15, 'content': 0.15
            }),
            'risk_thresholds': getattr(self, 'risk_thresholds', {
                'critical': 0.8, 'high': 0.6, 'medium': 0.4
            })
        }
    
    def reset_to_defaults(self) -> bool:
        """
        Reset all configurations to default values.
        
        Returns:
            True if reset was successful
        """
        self.country_risk_scores = self._get_country_risk_scores()
        self.industry_risk_scores = self._get_industry_risk_scores()
        
        # Remove custom weights and thresholds
        if hasattr(self, 'risk_weights'):
            delattr(self, 'risk_weights')
        if hasattr(self, 'risk_thresholds'):
            delattr(self, 'risk_thresholds')
        
        logger.info("Reset all configurations to defaults")
        return True
    
    def bulk_update_countries(self, country_updates: Dict[str, float]) -> Dict[str, bool]:
        """
        Allow agents to update multiple country risk scores at once.
        
        Args:
            country_updates: Dictionary of country -> risk_score
            
        Returns:
            Dictionary showing success/failure for each update
        """
        results = {}
        for country, risk_score in country_updates.items():
            results[country] = self.update_country_risk_score(country, risk_score)
        
        successful_updates = sum(1 for success in results.values() if success)
        logger.info(f"Bulk country update: {successful_updates}/{len(country_updates)} successful")
        return results
    
    def bulk_update_industries(self, industry_updates: Dict[str, float]) -> Dict[str, bool]:
        """
        Allow agents to update multiple industry risk scores at once.
        
        Args:
            industry_updates: Dictionary of industry -> risk_score
            
        Returns:
            Dictionary showing success/failure for each update
        """
        results = {}
        for industry, risk_score in industry_updates.items():
            results[industry] = self.update_industry_risk_score(industry, risk_score)
        
        successful_updates = sum(1 for success in results.values() if success)
        logger.info(f"Bulk industry update: {successful_updates}/{len(industry_updates)} successful")
        return results