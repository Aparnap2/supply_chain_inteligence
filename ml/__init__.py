"""
Simple ML prediction service for supply chain risk assessment.
Agent-configurable risk scoring that can be called when needed.
"""

from .model_training import SimpleMLPredictor

# Agent-friendly ML service factory
def create_ml_predictor() -> SimpleMLPredictor:
    """
    Create a simple ML predictor for agents to use.
    
    Returns:
        SimpleMLPredictor instance ready for agent use
    """
    return SimpleMLPredictor()

# Agent configuration helper
def get_configurable_options() -> dict:
    """
    Get all configuration options that agents can modify.
    
    Returns:
        Dictionary with configurable options and descriptions
    """
    return {
        'country_risk_scores': 'Update risk scores for countries (0.0-1.0)',
        'industry_risk_scores': 'Update risk scores for industries (0.0-1.0)', 
        'risk_weights': 'Update calculation weights for risk factors',
        'risk_thresholds': 'Update thresholds for risk level classification',
        'available_methods': [
            'predict_supplier_risk(supplier, scraped_data=None)',
            'batch_predict(suppliers, scraped_data_map=None)',
            'get_risk_summary(suppliers, scraped_data_map=None)',
            'update_country_risk_score(country, risk_score)',
            'update_industry_risk_score(industry, risk_score)',
            'update_risk_weights(weights)',
            'update_risk_thresholds(thresholds)',
            'get_current_config()',
            'reset_to_defaults()'
        ]
    }

__all__ = [
    'SimpleMLPredictor',
    'create_ml_predictor',
    'get_configurable_options'
]