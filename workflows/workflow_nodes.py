"""
LangGraph Workflow Nodes Implementation

This module implements all the workflow nodes for the supply chain intelligence platform,
including data collection, ML training, agent coordination, and reporting.

Requirements covered: 5.3, 5.5
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state_management import CompleteWorkflowState, update_workflow_state
from .workflow_state_manager import WorkflowStateManager
from data_collection.advanced_collector import AdvancedDataCollector
from data_collection.api_collector import APIDataCollector
from ml.model_training import SimpleMLPredictor
from agents.coordination import AgentCoordinator
from agents.task_definitions import SupplyChainTaskDefinitions
from models.supplier import Supplier
from models.scraped_data import ScrapedData
from models.analysis_result import AnalysisResult


logger = logging.getLogger(__name__)


class SupplyChainWorkflowNodes:
    """
    Implementation of all workflow nodes for the supply chain intelligence platform.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 state_manager: Optional[WorkflowStateManager] = None):
        """
        Initialize workflow nodes with required components.
        
        Args:
            api_key: OpenAI API key for LLM operations
            state_manager: Workflow state manager instance
        """
        self.api_key = api_key
        self.state_manager = state_manager or WorkflowStateManager()
        
        # Initialize components
        self.ml_predictor = SimpleMLPredictor()
        self.agent_coordinator = AgentCoordinator(verbose=True)
        
        # Initialize task definitions with agent factory
        from agents.specialized_agents import SupplyChainAgents
        agent_factory = SupplyChainAgents()
        self.task_definitions = SupplyChainTaskDefinitions(agent_factory)
        
        logger.info("Initialized SupplyChainWorkflowNodes")
    
    async def initialize_workflow(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Initialize workflow with setup and validation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        workflow_id = state["workflow_id"]
        
        logger.info(f"Initializing workflow {workflow_id}")
        
        try:
            # Validate input suppliers
            suppliers = state.get("suppliers", [])
            if not suppliers:
                raise ValueError("No suppliers provided for analysis")
            
            # Validate analysis configuration
            config = state.get("analysis_config", {})
            if not config:
                logger.warning("No analysis configuration provided, using defaults")
                config = {
                    "enable_web_scraping": True,
                    "enable_api_collection": True,
                    "enable_ml_training": True,
                    "enable_crew_analysis": True,
                    "max_retries": 3
                }
            
            # Convert supplier dictionaries to Supplier objects if needed
            validated_suppliers = []
            for supplier_data in suppliers:
                if isinstance(supplier_data, dict):
                    supplier = Supplier(**supplier_data)
                else:
                    supplier = supplier_data
                validated_suppliers.append(supplier)
            
            # Update state with validated data
            duration_ms = (time.time() - start_time) * 1000
            
            updates = {
                "current_step": "collect_web_data",
                "initialization_complete": True,
                "suppliers": [supplier.model_dump() for supplier in validated_suppliers],
                "analysis_config": config,
                "processing_times": {"initialize_workflow": duration_ms}
            }
            
            updated_state = update_workflow_state(state, updates)
            
            logger.info(f"Workflow {workflow_id} initialized successfully with {len(validated_suppliers)} suppliers")
            return updated_state
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow {workflow_id}: {str(e)}")
            
            updates = {
                "current_step": "error_handling",
                "has_errors": True,
                "error_message": f"Initialization failed: {str(e)}",
                "failed_steps": state.get("failed_steps", []) + ["initialize_workflow"]
            }
            
            return update_workflow_state(state, updates, validate=False)
    
    async def collect_web_data(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Collect web data using Crawl4AI integration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        workflow_id = state["workflow_id"]
        
        logger.info(f"Starting web data collection for workflow {workflow_id}")
        
        try:
            # Get suppliers from state
            supplier_dicts = state.get("suppliers", [])
            suppliers = [Supplier(**s) for s in supplier_dicts]
            
            # Check if web scraping is enabled
            config = state.get("analysis_config", {})
            if not config.get("enable_web_scraping", True):
                logger.info("Web scraping disabled, skipping web data collection")
                
                updates = {
                    "current_step": "collect_api_data",
                    "data_collection_status": "web_complete",
                    "scraped_data": []
                }
                return update_workflow_state(state, updates)
            
            # Collect web data using AdvancedDataCollector
            scraped_data = []
            async with AdvancedDataCollector(api_key=self.api_key) as collector:
                scraped_data = await collector.scrape_supplier_websites(suppliers)
            
            # Convert ScrapedData objects to dictionaries
            scraped_data_dicts = [data.model_dump() for data in scraped_data]
            
            # Calculate quality metrics
            successful_scrapes = [d for d in scraped_data if d.processing_status == "processed"]
            quality_score = len(successful_scrapes) / len(scraped_data) if scraped_data else 0.0
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Update state
            updates = {
                "current_step": "collect_api_data",
                "data_collection_status": "web_complete",
                "scraped_data": scraped_data_dicts,
                "data_quality_scores": {
                    **state.get("data_quality_scores", {}),
                    "web_scraping": quality_score
                },
                "processing_times": {
                    **state.get("processing_times", {}),
                    "collect_web_data": duration_ms
                }
            }
            
            updated_state = update_workflow_state(state, updates)
            
            logger.info(f"Web data collection completed for workflow {workflow_id}: "
                       f"{len(scraped_data)} records collected, quality score: {quality_score:.2f}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Web data collection failed for workflow {workflow_id}: {str(e)}")
            
            updates = {
                "current_step": "collect_api_data",  # Continue to next step
                "data_collection_status": "web_error",
                "has_errors": True,
                "error_message": f"Web data collection failed: {str(e)}",
                "failed_steps": state.get("failed_steps", []) + ["collect_web_data"],
                "scraped_data": []  # Empty data on failure
            }
            
            return update_workflow_state(state, updates, validate=False)
    
    async def collect_api_data(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Collect external API data for risk assessment.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        workflow_id = state["workflow_id"]
        
        logger.info(f"Starting API data collection for workflow {workflow_id}")
        
        try:
            # Get suppliers from state
            supplier_dicts = state.get("suppliers", [])
            suppliers = [Supplier(**s) for s in supplier_dicts]
            
            # Check if API collection is enabled
            config = state.get("analysis_config", {})
            if not config.get("enable_api_collection", True):
                logger.info("API collection disabled, skipping API data collection")
                
                updates = {
                    "current_step": "train_ml_models",
                    "data_collection_status": "complete",
                    "api_data": {}
                }
                return update_workflow_state(state, updates)
            
            # Collect API data
            api_collector = APIDataCollector()
            api_data = {}
            
            # Collect economic indicators
            try:
                economic_data = await api_collector.collect_economic_indicators(suppliers)
                api_data["economic_indicators"] = economic_data
            except Exception as e:
                logger.warning(f"Failed to collect economic indicators: {str(e)}")
                api_data["economic_indicators"] = {}
            
            # Collect weather data
            try:
                weather_data = await api_collector.collect_weather_risk_data(suppliers)
                api_data["weather_data"] = weather_data
            except Exception as e:
                logger.warning(f"Failed to collect weather data: {str(e)}")
                api_data["weather_data"] = {}
            
            # Collect trade data
            try:
                trade_data = await api_collector.collect_trade_data(suppliers)
                api_data["trade_data"] = trade_data
            except Exception as e:
                logger.warning(f"Failed to collect trade data: {str(e)}")
                api_data["trade_data"] = {}
            
            # Calculate quality score
            successful_collections = sum(1 for data in api_data.values() if data)
            quality_score = successful_collections / len(api_data) if api_data else 0.0
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Update state
            updates = {
                "current_step": "train_ml_models",
                "data_collection_status": "complete",
                "api_data": api_data,
                "data_quality_scores": {
                    **state.get("data_quality_scores", {}),
                    "api_collection": quality_score
                },
                "processing_times": {
                    **state.get("processing_times", {}),
                    "collect_api_data": duration_ms
                }
            }
            
            updated_state = update_workflow_state(state, updates)
            
            logger.info(f"API data collection completed for workflow {workflow_id}: "
                       f"{len(api_data)} data sources, quality score: {quality_score:.2f}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"API data collection failed for workflow {workflow_id}: {str(e)}")
            
            updates = {
                "current_step": "train_ml_models",  # Continue to next step
                "data_collection_status": "api_error",
                "has_errors": True,
                "error_message": f"API data collection failed: {str(e)}",
                "failed_steps": state.get("failed_steps", []) + ["collect_api_data"],
                "api_data": {}  # Empty data on failure
            }
            
            return update_workflow_state(state, updates, validate=False)
    
    async def train_ml_models(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Train ML models with collected data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        workflow_id = state["workflow_id"]
        
        logger.info(f"Starting ML model training for workflow {workflow_id}")
        
        try:
            # Check if ML training is enabled
            config = state.get("analysis_config", {})
            if not config.get("enable_ml_training", True):
                logger.info("ML training disabled, using rule-based approach only")
                
                updates = {
                    "current_step": "generate_ml_predictions",
                    "ml_training_complete": True,
                    "ml_models_metadata": {
                        "approach": "rule_based_only",
                        "training_skipped": True
                    }
                }
                return update_workflow_state(state, updates)
            
            # Get suppliers and historical data (if available)
            supplier_dicts = state.get("suppliers", [])
            suppliers = [Supplier(**s) for s in supplier_dicts]
            
            # For now, use simple training approach
            # In a real implementation, you would have historical risk events
            training_result = self.ml_predictor.train_simple_models(suppliers, [])
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Update state
            updates = {
                "current_step": "generate_ml_predictions",
                "ml_training_complete": True,
                "ml_models_metadata": {
                    **training_result,
                    "model_version": self.ml_predictor.model_version,
                    "training_duration_ms": duration_ms
                },
                "processing_times": {
                    **state.get("processing_times", {}),
                    "train_ml_models": duration_ms
                }
            }
            
            updated_state = update_workflow_state(state, updates)
            
            logger.info(f"ML model training completed for workflow {workflow_id}: "
                       f"Status: {training_result.get('status', 'unknown')}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"ML model training failed for workflow {workflow_id}: {str(e)}")
            
            updates = {
                "current_step": "generate_ml_predictions",  # Continue with rule-based approach
                "ml_training_complete": False,
                "has_errors": True,
                "error_message": f"ML training failed: {str(e)}",
                "failed_steps": state.get("failed_steps", []) + ["train_ml_models"],
                "ml_models_metadata": {
                    "approach": "rule_based_fallback",
                    "training_failed": True,
                    "error": str(e)
                }
            }
            
            return update_workflow_state(state, updates, validate=False)
    
    async def generate_ml_predictions(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Generate ML predictions for risk assessment.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        workflow_id = state["workflow_id"]
        
        logger.info(f"Starting ML prediction generation for workflow {workflow_id}")
        
        try:
            # Get suppliers and scraped data
            supplier_dicts = state.get("suppliers", [])
            suppliers = [Supplier(**s) for s in supplier_dicts]
            
            scraped_data_dicts = state.get("scraped_data", [])
            scraped_data = [ScrapedData(**d) for d in scraped_data_dicts]
            
            # Create mapping of supplier_id to scraped data
            scraped_data_map = {}
            for data in scraped_data:
                if data.supplier_id not in scraped_data_map:
                    scraped_data_map[data.supplier_id] = []
                scraped_data_map[data.supplier_id].append(data)
            
            # Generate predictions
            predictions = self.ml_predictor.batch_predict(suppliers, scraped_data_map)
            
            # Convert predictions to dictionaries
            prediction_dicts = [pred.model_dump() for pred in predictions]
            
            # Calculate average confidence
            avg_confidence = sum(pred.confidence for pred in predictions) / len(predictions) if predictions else 0.0
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Update state
            updates = {
                "current_step": "run_crew_analysis",
                "ml_predictions_complete": True,
                "ml_predictions": prediction_dicts,
                "data_quality_scores": {
                    **state.get("data_quality_scores", {}),
                    "ml_predictions": avg_confidence
                },
                "processing_times": {
                    **state.get("processing_times", {}),
                    "generate_ml_predictions": duration_ms
                }
            }
            
            updated_state = update_workflow_state(state, updates)
            
            logger.info(f"ML prediction generation completed for workflow {workflow_id}: "
                       f"{len(predictions)} predictions, avg confidence: {avg_confidence:.2f}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"ML prediction generation failed for workflow {workflow_id}: {str(e)}")
            
            updates = {
                "current_step": "run_crew_analysis",  # Continue to next step
                "ml_predictions_complete": False,
                "has_errors": True,
                "error_message": f"ML prediction generation failed: {str(e)}",
                "failed_steps": state.get("failed_steps", []) + ["generate_ml_predictions"],
                "ml_predictions": []  # Empty predictions on failure
            }
            
            return update_workflow_state(state, updates, validate=False)
    
    async def run_crew_analysis(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Run multi-agent analysis using CrewAI.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        workflow_id = state["workflow_id"]
        
        logger.info(f"Starting crew analysis for workflow {workflow_id}")
        
        try:
            # Check if crew analysis is enabled
            config = state.get("analysis_config", {})
            if not config.get("enable_crew_analysis", True):
                logger.info("Crew analysis disabled, skipping multi-agent analysis")
                
                updates = {
                    "current_step": "validate_and_structure",
                    "crew_analysis_complete": True,
                    "crew_analysis": {
                        "status": "skipped",
                        "message": "Crew analysis disabled in configuration"
                    }
                }
                return update_workflow_state(state, updates)
            
            # Create coordinated crew
            crew = self.agent_coordinator.create_coordinated_crew()
            
            # Create tasks for analysis
            tasks = self.task_definitions.create_complete_analysis_tasks()
            
            # Prepare inputs for crew execution
            crew_inputs = {
                "suppliers": state.get("suppliers", []),
                "scraped_data": state.get("scraped_data", []),
                "api_data": state.get("api_data", {}),
                "ml_predictions": state.get("ml_predictions", []),
                "analysis_config": state.get("analysis_config", {})
            }
            
            # Execute crew analysis
            result = self.agent_coordinator.execute_with_coordination(
                crew=crew,
                tasks=tasks,
                inputs=crew_inputs,
                context_sharing=True
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Update state
            updates = {
                "current_step": "validate_and_structure",
                "crew_analysis_complete": True,
                "crew_analysis": {
                    "execution_id": result.get("execution_id"),
                    "success": result.get("success", False),
                    "result": result.get("result"),
                    "execution_time": result.get("execution_time", 0),
                    "metrics": result.get("metrics", {})
                },
                "processing_times": {
                    **state.get("processing_times", {}),
                    "run_crew_analysis": duration_ms
                }
            }
            
            updated_state = update_workflow_state(state, updates)
            
            logger.info(f"Crew analysis completed for workflow {workflow_id}: "
                       f"Success: {result.get('success', False)}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Crew analysis failed for workflow {workflow_id}: {str(e)}")
            
            updates = {
                "current_step": "validate_and_structure",  # Continue to next step
                "crew_analysis_complete": False,
                "has_errors": True,
                "error_message": f"Crew analysis failed: {str(e)}",
                "failed_steps": state.get("failed_steps", []) + ["run_crew_analysis"],
                "crew_analysis": {
                    "status": "failed",
                    "error": str(e)
                }
            }
            
            return update_workflow_state(state, updates, validate=False)
    
    async def validate_and_structure(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Validate and structure results using Pydantic models.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        workflow_id = state["workflow_id"]
        
        logger.info(f"Starting validation and structuring for workflow {workflow_id}")
        
        try:
            # Reconstruct objects from state
            supplier_dicts = state.get("suppliers", [])
            suppliers = [Supplier(**s) for s in supplier_dicts]
            
            scraped_data_dicts = state.get("scraped_data", [])
            scraped_data = [ScrapedData(**d) for d in scraped_data_dicts]
            
            # Create analysis result
            analysis_result = AnalysisResult(
                analysis_id=workflow_id,
                suppliers=suppliers,
                scraped_data=scraped_data,
                risk_events=[],  # Would be populated from ML predictions
                ml_predictions=[],  # Would be populated from predictions
                overall_risk_score=self._calculate_overall_risk_score(state),
                recommendations=self._generate_recommendations(state),
                data_quality_score=self._calculate_overall_quality_score(state),
                analysis_metadata={
                    "processing_times": state.get("processing_times", {}),
                    "data_quality_scores": state.get("data_quality_scores", {}),
                    "ml_models_metadata": state.get("ml_models_metadata", {}),
                    "crew_analysis_metadata": state.get("crew_analysis", {})
                }
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Update state
            updates = {
                "current_step": "generate_final_report",
                "validation_complete": True,
                "validated_results": analysis_result.model_dump(),
                "processing_times": {
                    **state.get("processing_times", {}),
                    "validate_and_structure": duration_ms
                }
            }
            
            updated_state = update_workflow_state(state, updates)
            
            logger.info(f"Validation and structuring completed for workflow {workflow_id}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Validation and structuring failed for workflow {workflow_id}: {str(e)}")
            
            updates = {
                "current_step": "generate_final_report",  # Continue to final report
                "validation_complete": False,
                "has_errors": True,
                "error_message": f"Validation failed: {str(e)}",
                "failed_steps": state.get("failed_steps", []) + ["validate_and_structure"],
                "validated_results": None
            }
            
            return update_workflow_state(state, updates, validate=False)
    
    async def generate_final_report(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        """
        Generate comprehensive final report.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        workflow_id = state["workflow_id"]
        
        logger.info(f"Generating final report for workflow {workflow_id}")
        
        try:
            # Create comprehensive final report
            final_report = {
                "workflow_id": workflow_id,
                "generated_at": datetime.now().isoformat(),
                "execution_summary": {
                    "total_duration_ms": sum(state.get("processing_times", {}).values()),
                    "steps_completed": self._get_completed_steps(state),
                    "steps_failed": state.get("failed_steps", []),
                    "overall_success": not state.get("has_errors", False)
                },
                "data_summary": {
                    "suppliers_analyzed": len(state.get("suppliers", [])),
                    "web_data_collected": len(state.get("scraped_data", [])),
                    "api_data_sources": len(state.get("api_data", {})),
                    "ml_predictions_generated": len(state.get("ml_predictions", [])),
                    "overall_data_quality": self._calculate_overall_quality_score(state)
                },
                "risk_assessment": {
                    "overall_risk_score": self._calculate_overall_risk_score(state),
                    "high_risk_suppliers": self._identify_high_risk_suppliers(state),
                    "key_risk_factors": self._extract_key_risk_factors(state)
                },
                "detailed_results": {
                    "suppliers": state.get("suppliers", []),
                    "scraped_data": state.get("scraped_data", []),
                    "api_data": state.get("api_data", {}),
                    "ml_predictions": state.get("ml_predictions", []),
                    "crew_analysis": state.get("crew_analysis", {}),
                    "validated_results": state.get("validated_results")
                },
                "recommendations": self._generate_recommendations(state),
                "metadata": {
                    "processing_times": state.get("processing_times", {}),
                    "data_quality_scores": state.get("data_quality_scores", {}),
                    "ml_models_metadata": state.get("ml_models_metadata", {}),
                    "analysis_config": state.get("analysis_config", {})
                }
            }
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Update state to completed
            updates = {
                "current_step": "completed",
                "analysis_complete": True,
                "final_report": final_report,
                "processing_times": {
                    **state.get("processing_times", {}),
                    "generate_final_report": duration_ms
                }
            }
            
            updated_state = update_workflow_state(state, updates)
            
            logger.info(f"Final report generated for workflow {workflow_id}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Final report generation failed for workflow {workflow_id}: {str(e)}")
            
            # Create minimal error report
            error_report = {
                "workflow_id": workflow_id,
                "generated_at": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
                "partial_results": {
                    "suppliers": state.get("suppliers", []),
                    "processing_times": state.get("processing_times", {}),
                    "failed_steps": state.get("failed_steps", [])
                }
            }
            
            updates = {
                "current_step": "error_handling",
                "analysis_complete": False,
                "has_errors": True,
                "error_message": f"Final report generation failed: {str(e)}",
                "failed_steps": state.get("failed_steps", []) + ["generate_final_report"],
                "final_report": error_report
            }
            
            return update_workflow_state(state, updates, validate=False)
    
    def _calculate_overall_risk_score(self, state: CompleteWorkflowState) -> float:
        """Calculate overall risk score from predictions."""
        predictions = state.get("ml_predictions", [])
        if not predictions:
            return 50.0  # Default neutral score
        
        # Convert risk levels to numeric scores
        risk_scores = []
        for pred in predictions:
            risk_level = pred.get("prediction", "medium")
            score_map = {"low": 25, "medium": 50, "high": 75, "critical": 90}
            risk_scores.append(score_map.get(risk_level, 50))
        
        return sum(risk_scores) / len(risk_scores)
    
    def _calculate_overall_quality_score(self, state: CompleteWorkflowState) -> float:
        """Calculate overall data quality score."""
        quality_scores = state.get("data_quality_scores", {})
        if not quality_scores:
            return 0.5  # Default neutral score
        
        return sum(quality_scores.values()) / len(quality_scores)
    
    def _identify_high_risk_suppliers(self, state: CompleteWorkflowState) -> List[str]:
        """Identify suppliers with high or critical risk levels."""
        predictions = state.get("ml_predictions", [])
        high_risk = []
        
        for pred in predictions:
            if pred.get("prediction") in ["high", "critical"]:
                high_risk.append(pred.get("supplier_id"))
        
        return high_risk
    
    def _extract_key_risk_factors(self, state: CompleteWorkflowState) -> List[str]:
        """Extract key risk factors from analysis."""
        risk_factors = []
        
        # From scraped data
        scraped_data = state.get("scraped_data", [])
        for data in scraped_data:
            risk_indicators = data.get("risk_indicators", [])
            risk_factors.extend(risk_indicators)
        
        # From ML predictions
        predictions = state.get("ml_predictions", [])
        for pred in predictions:
            feature_importance = pred.get("feature_importance", {})
            # Get top risk factors
            top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            risk_factors.extend([factor[0] for factor in top_factors])
        
        # Return unique factors
        return list(set(risk_factors))
    
    def _generate_recommendations(self, state: CompleteWorkflowState) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Based on high-risk suppliers
        high_risk_suppliers = self._identify_high_risk_suppliers(state)
        if high_risk_suppliers:
            recommendations.append(
                f"Immediate attention required for {len(high_risk_suppliers)} high-risk suppliers"
            )
        
        # Based on data quality
        quality_score = self._calculate_overall_quality_score(state)
        if quality_score < 0.7:
            recommendations.append(
                "Improve data collection processes to enhance analysis accuracy"
            )
        
        # Based on failed steps
        failed_steps = state.get("failed_steps", [])
        if failed_steps:
            recommendations.append(
                f"Review and address failures in: {', '.join(failed_steps)}"
            )
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "Continue monitoring supplier risk indicators",
                "Maintain regular data collection and analysis cycles",
                "Review and update risk assessment criteria periodically"
            ])
        
        return recommendations
    
    def _get_completed_steps(self, state: CompleteWorkflowState) -> List[str]:
        """Get list of successfully completed steps."""
        completed = []
        
        if state.get("initialization_complete"):
            completed.append("initialize_workflow")
        if state.get("data_collection_status") in ["web_complete", "complete"]:
            completed.append("collect_web_data")
        if state.get("data_collection_status") in ["api_complete", "complete"]:
            completed.append("collect_api_data")
        if state.get("ml_training_complete"):
            completed.append("train_ml_models")
        if state.get("ml_predictions_complete"):
            completed.append("generate_ml_predictions")
        if state.get("crew_analysis_complete"):
            completed.append("run_crew_analysis")
        if state.get("validation_complete"):
            completed.append("validate_and_structure")
        if state.get("analysis_complete"):
            completed.append("generate_final_report")
        
        return completed