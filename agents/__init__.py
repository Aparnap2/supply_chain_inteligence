"""
CrewAI Multi-Agent System for Supply Chain Intelligence Platform

This package implements a comprehensive multi-agent system using CrewAI for
supply chain risk analysis and strategic recommendation generation.

Components:
- Specialized AI Agents: Data Analysis, ML Specialist, Risk Assessment, Strategy
- Agent Coordination: Task management, delegation, and performance monitoring
- Task Definitions: Structured tasks with Pydantic output models
- Crew Orchestrator: Main interface for running complete analysis workflows

Requirements covered: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from .specialized_agents import (
    SupplyChainAgents,
    create_data_analysis_agent,
    create_ml_specialist_agent,
    create_risk_assessment_agent,
    create_strategy_agent
)

from .coordination import (
    AgentCoordinator,
    create_coordinated_crew
)

from .task_definitions import (
    SupplyChainTaskDefinitions,
    DataAnalysisOutput,
    MLValidationOutput,
    RiskAssessmentOutput,
    StrategyOutput,
    create_data_analysis_task,
    create_ml_validation_task,
    create_risk_assessment_task,
    create_strategy_generation_task
)

from .crew_orchestrator import (
    SupplyChainCrewOrchestrator,
    run_supply_chain_analysis
)

__all__ = [
    # Agent creation
    "SupplyChainAgents",
    "create_data_analysis_agent",
    "create_ml_specialist_agent", 
    "create_risk_assessment_agent",
    "create_strategy_agent",
    
    # Coordination
    "AgentCoordinator",
    "create_coordinated_crew",
    
    # Task definitions
    "SupplyChainTaskDefinitions",
    "DataAnalysisOutput",
    "MLValidationOutput", 
    "RiskAssessmentOutput",
    "StrategyOutput",
    "create_data_analysis_task",
    "create_ml_validation_task",
    "create_risk_assessment_task", 
    "create_strategy_generation_task",
    
    # Main orchestrator
    "SupplyChainCrewOrchestrator",
    "run_supply_chain_analysis"
]