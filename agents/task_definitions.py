"""
Agent Task Definitions for Supply Chain Intelligence Platform

This module implements specific task definitions for each specialized agent,
including structured output requirements and confidence assessment.

Requirements covered: 3.6
"""

from crewai import Task
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

from .specialized_agents import SupplyChainAgents


# Pydantic models for structured task outputs
class DataAnalysisOutput(BaseModel):
    """Structured output for data analysis tasks"""
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    patterns_identified: List[str] = Field(
        default_factory=list, 
        description="List of significant patterns found in the data"
    )
    anomalies_detected: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Anomalies with severity and description"
    )
    risk_indicators: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Risk indicators with confidence scores"
    )
    data_quality_score: float = Field(
        ..., ge=0, le=1, 
        description="Overall data quality score (0-1)"
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights and findings from the analysis"
    )
    confidence_level: float = Field(
        ..., ge=0, le=1,
        description="Confidence level in the analysis results"
    )


class MLValidationOutput(BaseModel):
    """Structured output for ML validation tasks"""
    validation_id: str = Field(..., description="Unique identifier for this validation")
    model_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Model performance metrics (accuracy, precision, recall, etc.)"
    )
    prediction_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for each prediction category"
    )
    feature_importance: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Feature importance rankings with scores"
    )
    model_reliability: Literal["high", "medium", "low"] = Field(
        ..., description="Overall model reliability assessment"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for model improvement"
    )
    validation_summary: str = Field(
        ..., description="Summary of validation findings"
    )


class RiskAssessmentOutput(BaseModel):
    """Structured output for risk assessment tasks"""
    assessment_id: str = Field(..., description="Unique identifier for this assessment")
    risk_categories: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Risks categorized by type (geopolitical, financial, etc.)"
    )
    severity_levels: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of risks by severity level"
    )
    business_impact_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Assessment of potential business impact"
    )
    likelihood_timelines: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Timeline predictions for risk materialization"
    )
    priority_matrix: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Risk prioritization matrix with impact-probability scores"
    )
    critical_risks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Highest priority risks requiring immediate attention"
    )


class StrategyOutput(BaseModel):
    """Structured output for strategy generation tasks"""
    strategy_id: str = Field(..., description="Unique identifier for this strategy")
    actionable_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Specific actionable recommendations with implementation details"
    )
    mitigation_strategies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Risk mitigation strategies with timelines and resources"
    )
    implementation_roadmap: Dict[str, Any] = Field(
        default_factory=dict,
        description="Phased implementation plan with milestones"
    )
    cost_benefit_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cost-benefit analysis for recommended strategies"
    )
    success_metrics: List[str] = Field(
        default_factory=list,
        description="Key performance indicators for strategy success"
    )
    executive_summary: str = Field(
        ..., description="Executive summary of strategic recommendations"
    )


class SupplyChainTaskDefinitions:
    """
    Factory class for creating specialized tasks for supply chain analysis agents
    
    Requirements: 3.6 - Create task definitions with structured output requirements
    """
    
    def __init__(self, agent_factory: SupplyChainAgents):
        """
        Initialize task definitions with agent factory
        
        Args:
            agent_factory: Factory for creating specialized agents
        """
        self.agent_factory = agent_factory
    
    def create_data_analysis_task(
        self, 
        suppliers_data: List[Dict[str, Any]], 
        scraped_data: List[Dict[str, Any]],
        context_data: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create data analysis task with structured output requirements
        
        Requirements: 3.6 - Create data analysis tasks with structured output requirements
        
        Args:
            suppliers_data: Supplier information and metrics
            scraped_data: Web-scraped content and news data
            context_data: Additional context from previous analysis
            
        Returns:
            Configured Task for data analysis
        """
        description = f"""
        Analyze the provided supply chain data to identify patterns, anomalies, and risk indicators.
        
        **Data Sources:**
        - {len(suppliers_data)} supplier records with financial and operational metrics
        - {len(scraped_data)} web-scraped content items including news and company information
        - Additional context: {bool(context_data)}
        
        **Analysis Requirements:**
        1. **Pattern Recognition:**
           - Identify trends in supplier performance metrics
           - Detect correlations between external factors and supplier behavior
           - Analyze sentiment patterns in news and web content
        
        2. **Anomaly Detection:**
           - Flag unusual changes in financial health indicators
           - Identify outliers in operational performance
           - Detect suspicious patterns in web content or news sentiment
        
        3. **Risk Indicator Identification:**
           - Extract risk signals from financial data
           - Identify geographic or industry-specific risk concentrations
           - Assess data quality and completeness issues
        
        4. **Quality Assessment:**
           - Evaluate data completeness and accuracy
           - Assign quality scores to different data sources
           - Identify gaps that may impact analysis reliability
        
        **Output Requirements:**
        - Provide structured analysis with confidence scores
        - Include specific examples and evidence for each finding
        - Prioritize findings by potential business impact
        """
        
        expected_output = """
        A comprehensive data analysis report with:
        - Identified patterns with statistical significance
        - Detected anomalies with severity classifications
        - Risk indicators with confidence assessments
        - Data quality evaluation and recommendations
        - Key insights prioritized by business impact
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.agent_factory.create_data_analysis_agent(),
            output_pydantic=DataAnalysisOutput
        )
    
    def create_ml_validation_task(
        self,
        ml_predictions: List[Dict[str, Any]],
        model_metadata: Dict[str, Any],
        validation_data: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create ML validation task with confidence assessment
        
        Requirements: 3.6 - Implement ML validation tasks with confidence assessment
        
        Args:
            ml_predictions: ML model predictions to validate
            model_metadata: Information about the models used
            validation_data: Additional validation datasets
            
        Returns:
            Configured Task for ML validation
        """
        description = f"""
        Validate machine learning model predictions and assess confidence levels for supply chain risk assessment.
        
        **Validation Scope:**
        - {len(ml_predictions)} ML predictions across risk classification and impact scoring
        - Model metadata: {list(model_metadata.keys()) if model_metadata else 'None'}
        - Validation data available: {bool(validation_data)}
        
        **Validation Requirements:**
        1. **Model Performance Assessment:**
           - Evaluate classification accuracy for risk levels (low, medium, high, critical)
           - Assess regression performance for impact scores (0-10 scale)
           - Validate disruption probability predictions (0-1 scale)
        
        2. **Confidence Analysis:**
           - Calculate prediction confidence intervals
           - Identify low-confidence predictions requiring human review
           - Assess model uncertainty and reliability
        
        3. **Feature Importance Validation:**
           - Analyze feature importance rankings
           - Validate that important features align with domain knowledge
           - Identify potential model bias or overfitting issues
        
        4. **Model Reliability Assessment:**
           - Compare predictions against historical patterns
           - Assess model consistency across different supplier segments
           - Evaluate model performance degradation indicators
        
        **Quality Thresholds:**
        - Classification accuracy: minimum 80%
        - Regression R²: minimum 0.7
        - Confidence threshold: flag predictions below 70% confidence
        """
        
        expected_output = """
        A comprehensive ML validation report including:
        - Model performance metrics with statistical significance
        - Confidence assessments for each prediction category
        - Feature importance analysis with domain validation
        - Model reliability classification and recommendations
        - Specific recommendations for model improvement
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.agent_factory.create_ml_specialist_agent(),
            output_pydantic=MLValidationOutput
        )
    
    def create_risk_assessment_task(
        self,
        analysis_results: Dict[str, Any],
        ml_validation: Dict[str, Any],
        business_context: Dict[str, Any]
    ) -> Task:
        """
        Create risk assessment task with severity categorization
        
        Requirements: 3.6 - Develop risk assessment tasks with severity categorization
        
        Args:
            analysis_results: Results from data analysis
            ml_validation: ML validation results
            business_context: Business objectives and constraints
            
        Returns:
            Configured Task for risk assessment
        """
        description = f"""
        Categorize supply chain risks, evaluate business impact, and assess likelihood timelines based on analysis results.
        
        **Input Sources:**
        - Data analysis results: {bool(analysis_results)}
        - ML validation results: {bool(ml_validation)}
        - Business context: {list(business_context.keys()) if business_context else 'None'}
        
        **Risk Assessment Requirements:**
        1. **Risk Categorization:**
           - Geopolitical risks (trade wars, sanctions, political instability)
           - Financial risks (supplier bankruptcy, credit issues, currency fluctuation)
           - Environmental risks (natural disasters, climate change, resource scarcity)
           - Operational risks (capacity constraints, quality issues, logistics disruption)
           - Regulatory risks (compliance changes, new regulations, certification issues)
           - Cyber risks (data breaches, system failures, cyber attacks)
        
        2. **Severity Assessment:**
           - Critical: Immediate threat to business continuity (>$10M impact, <30 days)
           - High: Significant business impact (>$1M impact, <90 days)
           - Medium: Moderate impact with manageable consequences (<$1M, <180 days)
           - Low: Minor impact with minimal business disruption (<$100K, >180 days)
        
        3. **Business Impact Evaluation:**
           - Financial impact assessment (direct costs, opportunity costs, recovery costs)
           - Operational impact (production delays, quality issues, customer satisfaction)
           - Strategic impact (market position, competitive advantage, reputation)
           - Compliance impact (regulatory penalties, legal exposure)
        
        4. **Timeline Assessment:**
           - Immediate risks (0-30 days)
           - Short-term risks (1-6 months)
           - Medium-term risks (6-18 months)
           - Long-term risks (18+ months)
        
        5. **Priority Matrix:**
           - Create impact-probability matrix for risk prioritization
           - Identify critical risks requiring immediate attention
           - Develop risk monitoring and escalation criteria
        """
        
        expected_output = """
        A comprehensive risk assessment including:
        - Risks categorized by type with detailed descriptions
        - Severity levels with quantified impact assessments
        - Business impact analysis across financial, operational, and strategic dimensions
        - Timeline predictions with confidence intervals
        - Priority matrix with actionable risk rankings
        - Critical risks requiring immediate executive attention
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.agent_factory.create_risk_assessment_agent(),
            output_pydantic=RiskAssessmentOutput
        )
    
    def create_strategy_generation_task(
        self,
        risk_assessment: Dict[str, Any],
        business_objectives: Dict[str, Any],
        resource_constraints: Dict[str, Any]
    ) -> Task:
        """
        Create strategy generation task with actionable recommendations
        
        Requirements: 3.6 - Create strategy generation tasks with actionable recommendations
        
        Args:
            risk_assessment: Risk assessment results
            business_objectives: Business goals and priorities
            resource_constraints: Budget, timeline, and resource limitations
            
        Returns:
            Configured Task for strategy generation
        """
        description = f"""
        Generate actionable recommendations and comprehensive mitigation strategies based on risk assessment and business objectives.
        
        **Strategic Context:**
        - Risk assessment results: {bool(risk_assessment)}
        - Business objectives: {list(business_objectives.keys()) if business_objectives else 'None'}
        - Resource constraints: {list(resource_constraints.keys()) if resource_constraints else 'None'}
        
        **Strategy Development Requirements:**
        1. **Actionable Recommendations:**
           - Specific, measurable actions with clear outcomes
           - Implementation timelines with milestones and dependencies
           - Resource requirements (budget, personnel, technology)
           - Success criteria and key performance indicators
        
        2. **Risk Mitigation Strategies:**
           - Supplier diversification strategies
           - Contingency planning and business continuity measures
           - Supply chain resilience improvements
           - Risk monitoring and early warning systems
        
        3. **Implementation Roadmap:**
           - Phase 1 (0-3 months): Critical risk mitigation and immediate actions
           - Phase 2 (3-12 months): Strategic improvements and process optimization
           - Phase 3 (12+ months): Long-term resilience building and innovation
        
        4. **Cost-Benefit Analysis:**
           - Implementation costs vs. risk mitigation value
           - Return on investment calculations
           - Break-even analysis and payback periods
           - Sensitivity analysis for key assumptions
        
        5. **Strategic Alignment:**
           - Alignment with business objectives and priorities
           - Integration with existing supply chain strategies
           - Consideration of competitive implications
           - Stakeholder impact assessment
        
        **Deliverable Requirements:**
        - Executive-ready recommendations with clear business justification
        - Detailed implementation plans with resource allocation
        - Risk-adjusted financial projections
        - Change management and communication strategies
        """
        
        expected_output = """
        A comprehensive strategic plan including:
        - Prioritized actionable recommendations with implementation details
        - Risk mitigation strategies with timelines and resource requirements
        - Phased implementation roadmap with clear milestones
        - Cost-benefit analysis with financial projections
        - Success metrics and monitoring framework
        - Executive summary suitable for C-level presentation
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.agent_factory.create_strategy_agent(),
            output_pydantic=StrategyOutput
        )
    
    def create_complete_analysis_workflow(
        self,
        suppliers_data: List[Dict[str, Any]],
        scraped_data: List[Dict[str, Any]],
        ml_predictions: List[Dict[str, Any]],
        model_metadata: Dict[str, Any],
        business_context: Dict[str, Any]
    ) -> List[Task]:
        """
        Create a complete workflow of interconnected tasks
        
        Args:
            suppliers_data: Supplier information
            scraped_data: Web-scraped content
            ml_predictions: ML model predictions
            model_metadata: Model information
            business_context: Business objectives and constraints
            
        Returns:
            List of interconnected tasks for complete analysis
        """
        # Create individual tasks
        data_task = self.create_data_analysis_task(suppliers_data, scraped_data)
        ml_task = self.create_ml_validation_task(ml_predictions, model_metadata)
        risk_task = self.create_risk_assessment_task({}, {}, business_context)
        strategy_task = self.create_strategy_generation_task({}, business_context, {})
        
        # Set up task dependencies through context
        ml_task.context = [data_task]
        risk_task.context = [data_task, ml_task]
        strategy_task.context = [data_task, ml_task, risk_task]
        
        return [data_task, ml_task, risk_task, strategy_task]


# Convenience functions for creating individual tasks
def create_data_analysis_task(
    suppliers_data: List[Dict[str, Any]], 
    scraped_data: List[Dict[str, Any]],
    llm_model: str = "gpt-4o"
) -> Task:
    """Create a data analysis task"""
    agent_factory = SupplyChainAgents(llm_model)
    task_factory = SupplyChainTaskDefinitions(agent_factory)
    return task_factory.create_data_analysis_task(suppliers_data, scraped_data)


def create_ml_validation_task(
    ml_predictions: List[Dict[str, Any]],
    model_metadata: Dict[str, Any],
    llm_model: str = "gpt-4o"
) -> Task:
    """Create an ML validation task"""
    agent_factory = SupplyChainAgents(llm_model)
    task_factory = SupplyChainTaskDefinitions(agent_factory)
    return task_factory.create_ml_validation_task(ml_predictions, model_metadata)


def create_risk_assessment_task(
    analysis_results: Dict[str, Any],
    ml_validation: Dict[str, Any],
    business_context: Dict[str, Any],
    llm_model: str = "gpt-4o"
) -> Task:
    """Create a risk assessment task"""
    agent_factory = SupplyChainAgents(llm_model)
    task_factory = SupplyChainTaskDefinitions(agent_factory)
    return task_factory.create_risk_assessment_task(analysis_results, ml_validation, business_context)


def create_strategy_generation_task(
    risk_assessment: Dict[str, Any],
    business_objectives: Dict[str, Any],
    resource_constraints: Dict[str, Any],
    llm_model: str = "gpt-4o"
) -> Task:
    """Create a strategy generation task"""
    agent_factory = SupplyChainAgents(llm_model)
    task_factory = SupplyChainTaskDefinitions(agent_factory)
    return task_factory.create_strategy_generation_task(risk_assessment, business_objectives, resource_constraints)