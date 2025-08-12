"""
Specialized AI Agents for Supply Chain Intelligence Platform

This module implements the four specialized CrewAI agents required for comprehensive
supply chain risk analysis and strategic recommendation generation.

Requirements covered: 3.1, 3.2, 3.3, 3.4
"""

from crewai import Agent
from typing import List, Optional
import os


class SupplyChainAgents:
    """Factory class for creating specialized supply chain analysis agents"""
    
    def __init__(self, llm_model: str = "gpt-4o", verbose: bool = True):
        """
        Initialize the agent factory
        
        Args:
            llm_model: The LLM model to use for agents
            verbose: Whether to enable verbose logging
        """
        self.llm_model = llm_model
        self.verbose = verbose
    
    def create_data_analysis_agent(self) -> Agent:
        """
        Create Data Analysis Agent with data pattern recognition capabilities
        
        Requirements: 3.1 - Deploy Data Analysis Agent to identify patterns, 
        anomalies, and risk indicators in collected data
        """
        return Agent(
            role="Supply Chain Data Analyst",
            goal="Analyze collected supply chain data to identify patterns, anomalies, and risk indicators that could impact business operations",
            backstory="""You are an expert data analyst with deep knowledge of supply chain operations 
            and risk factors. You have extensive experience in pattern recognition, statistical analysis, 
            and identifying early warning signals in complex datasets. Your analytical skills help 
            organizations understand their supply chain vulnerabilities before they become critical issues.
            
            You excel at:
            - Identifying unusual patterns in supplier behavior and performance metrics
            - Detecting anomalies in financial health indicators and operational data
            - Recognizing correlations between external factors (weather, geopolitics, economics) and supply chain risks
            - Extracting meaningful insights from web-scraped content and news sentiment
            - Quantifying data quality and reliability for downstream analysis""",
            verbose=self.verbose,
            allow_delegation=False,  # Specialist focused on data analysis
            max_iter=3,
            memory=True
        )
    
    def create_ml_specialist_agent(self) -> Agent:
        """
        Create ML Specialist Agent for model validation and interpretation
        
        Requirements: 3.2 - Activate ML Specialist Agent to validate model 
        predictions and assess confidence levels
        """
        return Agent(
            role="Machine Learning Specialist",
            goal="Validate machine learning model predictions, assess confidence levels, and interpret model outputs for supply chain risk assessment",
            backstory="""You are a senior machine learning engineer with specialized expertise in 
            predictive modeling for supply chain and risk management applications. You have deep 
            understanding of model validation techniques, confidence assessment, and feature importance 
            analysis. Your role is critical in ensuring that ML predictions are reliable and actionable.
            
            Your expertise includes:
            - Validating classification models for risk level predictions (low, medium, high, critical)
            - Assessing regression model performance for impact scores and disruption probabilities
            - Interpreting feature importance and model decision boundaries
            - Identifying model drift and performance degradation
            - Providing confidence intervals and uncertainty quantification
            - Recommending model improvements and retraining strategies""",
            verbose=self.verbose,
            allow_delegation=False,  # Specialist focused on ML validation
            max_iter=3,
            memory=True
        )
    
    def create_risk_assessment_agent(self) -> Agent:
        """
        Create Risk Assessment Agent for risk categorization and prioritization
        
        Requirements: 3.3 - Engage Risk Assessment Agent to categorize risks, 
        evaluate business impact, and assess likelihood timelines
        """
        return Agent(
            role="Risk Assessment Expert",
            goal="Categorize supply chain risks, evaluate business impact, assess likelihood timelines, and prioritize risk mitigation efforts",
            backstory="""You are a senior risk management expert with extensive experience in global 
            supply chain risk assessment and business continuity planning. You have worked with 
            Fortune 500 companies to identify, categorize, and mitigate supply chain disruptions. 
            Your expertise spans multiple risk categories and business impact assessment methodologies.
            
            Your specializations include:
            - Categorizing risks by type (geopolitical, financial, environmental, operational, regulatory, cyber)
            - Assessing business impact severity and potential financial consequences
            - Evaluating risk likelihood and timeline predictions
            - Prioritizing risks based on impact-probability matrices
            - Developing risk mitigation strategies and contingency plans
            - Understanding regulatory compliance and industry-specific risk factors""",
            verbose=self.verbose,
            allow_delegation=True,  # Can coordinate with other agents for comprehensive assessment
            max_iter=3,
            memory=True
        )
    
    def create_strategy_agent(self) -> Agent:
        """
        Create Strategy Agent for recommendation generation
        
        Requirements: 3.4 - Deploy Strategy Agent to generate actionable 
        recommendations and mitigation strategies
        """
        return Agent(
            role="Strategic Supply Chain Advisor",
            goal="Generate actionable recommendations and comprehensive mitigation strategies based on risk analysis and business objectives",
            backstory="""You are a strategic consultant specializing in supply chain optimization 
            and risk mitigation. You have advised C-level executives at major corporations on 
            supply chain resilience, vendor diversification, and business continuity strategies. 
            Your recommendations are practical, cost-effective, and aligned with business objectives.
            
            Your strategic expertise covers:
            - Developing actionable mitigation strategies for identified risks
            - Recommending supplier diversification and contingency planning approaches
            - Proposing operational improvements and process optimizations
            - Balancing risk mitigation costs with potential impact savings
            - Creating implementation roadmaps with clear timelines and responsibilities
            - Aligning supply chain strategies with broader business objectives and constraints""",
            verbose=self.verbose,
            allow_delegation=False,  # Specialist focused on strategy generation
            max_iter=3,
            memory=True
        )
    
    def get_all_agents(self) -> List[Agent]:
        """
        Get all specialized agents for the supply chain analysis crew
        
        Returns:
            List of all four specialized agents
        """
        return [
            self.create_data_analysis_agent(),
            self.create_ml_specialist_agent(),
            self.create_risk_assessment_agent(),
            self.create_strategy_agent()
        ]
    
    def get_agent_roles(self) -> List[str]:
        """
        Get the roles of all specialized agents
        
        Returns:
            List of agent role names
        """
        return [
            "Supply Chain Data Analyst",
            "Machine Learning Specialist", 
            "Risk Assessment Expert",
            "Strategic Supply Chain Advisor"
        ]


# Convenience functions for direct agent creation
def create_data_analysis_agent(llm_model: str = "gpt-4o", verbose: bool = True) -> Agent:
    """Create and return a Data Analysis Agent"""
    factory = SupplyChainAgents(llm_model, verbose)
    return factory.create_data_analysis_agent()


def create_ml_specialist_agent(llm_model: str = "gpt-4o", verbose: bool = True) -> Agent:
    """Create and return an ML Specialist Agent"""
    factory = SupplyChainAgents(llm_model, verbose)
    return factory.create_ml_specialist_agent()


def create_risk_assessment_agent(llm_model: str = "gpt-4o", verbose: bool = True) -> Agent:
    """Create and return a Risk Assessment Agent"""
    factory = SupplyChainAgents(llm_model, verbose)
    return factory.create_risk_assessment_agent()


def create_strategy_agent(llm_model: str = "gpt-4o", verbose: bool = True) -> Agent:
    """Create and return a Strategy Agent"""
    factory = SupplyChainAgents(llm_model, verbose)
    return factory.create_strategy_agent()