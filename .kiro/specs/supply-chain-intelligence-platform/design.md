# Design Document

## Overview

The Supply Chain Intelligence & Risk Management Platform is a comprehensive AI-powered system that integrates multiple cutting-edge technologies to provide enterprise-level supply chain risk assessment and early warning capabilities. The platform combines workflow orchestration (LangGraph), multi-agent coordination (CrewAI), automated web scraping (Crawl4AI), machine learning pipelines (scikit-learn), and interactive visualization (Streamlit) to deliver actionable intelligence for supply chain risk management.

The system addresses the critical business problem where supply chain disruptions cost enterprises an average of $184M annually due to limited visibility and prediction capabilities. By automating data collection, applying machine learning for risk prediction, and providing AI-powered analysis through specialized agents, the platform enables proactive risk management and strategic decision-making.

## Architecture

### High-Level Architecture

The platform follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │   Streamlit     │    Plotly       │      Export APIs        │ │
│  │   Dashboard     │  Visualizations │   (JSON/CSV/PDF)        │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                   AI ORCHESTRATION LAYER                        │
│  ┌─────────────────┬─────────────────────────────────────────────┐ │
│  │   LangGraph     │                CrewAI                      │ │
│  │  - StateGraph   │  - Data Analysis Agent                     │ │
│  │  - Checkpoints  │  - ML Specialist Agent                     │ │
│  │  - Error Handling│  - Risk Assessment Agent                  │ │
│  │  - Conditional  │  - Strategy Agent                          │ │
│  │    Routing      │  - Task Delegation                         │ │
│  └─────────────────┴─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                  DATA PROCESSING LAYER                          │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │    Pandas       │     Pydantic    │        sklearn          │ │
│  │  Data Cleaning  │  Type Safety &  │   ML Models &           │ │
│  │  Transformation │   Validation    │   Predictions           │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                   DATA COLLECTION LAYER                         │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │   Crawl4AI      │   Free APIs     │   File Uploads          │ │
│  │   Web Scraping  │   Real-time     │   CSV/Excel             │ │
│  │   - Supplier    │   Data Feeds    │   Documents             │ │
│  │     Websites    │   - World Bank  │   - Manual Input        │ │
│  │   - News        │   - Weather     │   - Historical Data     │ │
│  │     Sources     │   - Trade Data  │                         │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                      PERSISTENCE LAYER                          │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │   SQLite        │   File System   │      Memory Cache       │ │
│  │  Checkpoints    │   Model Storage │   Session State         │ │
│  │  Workflow State │   Reports       │   Temporary Data        │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack Integration

**LangGraph Integration:**
- Provides workflow orchestration with StateGraph for managing complex multi-step processes
- Implements checkpointing with SqliteSaver for durable execution and recovery
- Handles conditional routing based on data collection and processing results
- Manages state transitions between data collection, ML training, agent analysis, and reporting phases

**CrewAI Integration:**
- Coordinates specialized AI agents with distinct roles and responsibilities
- Implements task delegation with `allow_delegation=True` for coordinating agents
- Uses sequential process execution for structured analysis workflow
- Provides collaborative analysis through multi-agent interaction

**Crawl4AI Integration:**
- Implements LLMExtractionStrategy for intelligent content parsing
- Uses JsonCssExtractionStrategy for structured data extraction
- Applies RegexExtractionStrategy for pattern-based data extraction
- Provides robust fallback strategies for reliable data collection

**Pydantic Integration:**
- Ensures type safety and data validation throughout the system
- Defines structured models for suppliers, risk events, and analysis results
- Provides automatic serialization/deserialization for data persistence
- Validates data quality and consistency across all processing stages

## Components and Interfaces

### Core Components

#### 1. Data Collection Engine
**Purpose:** Automated collection and integration of supply chain data from multiple sources

**Key Classes:**
- `AdvancedDataCollector`: Main orchestrator for data collection operations
- `ScrapedData`: Pydantic model for web-scraped content with metadata
- `Supplier`: Core supplier information model with validation

**Interfaces:**
```python
class DataCollectorInterface:
    async def scrape_supplier_websites(self, suppliers: List[Supplier]) -> List[ScrapedData]
    async def collect_api_data(self, suppliers: List[Supplier]) -> Dict[str, Any]
    async def process_file_uploads(self, files: List[UploadedFile]) -> List[Dict]
```

**Crawl4AI Integration:**
- Uses `WebCrawler` with `LLMExtractionStrategy` for intelligent content extraction
- Implements fallback strategies with `CosineStrategy` for content filtering
- Applies `JsonCssExtractionStrategy` for structured data extraction from supplier websites

#### 2. Machine Learning Pipeline
**Purpose:** Risk prediction and impact assessment using trained models

**Key Classes:**
- `SupplyChainMLPipeline`: Main ML orchestrator with model training and prediction
- `MLPrediction`: Pydantic model for ML prediction results with confidence scores
- `RiskEvent`: Structured risk event model with severity and probability

**Interfaces:**
```python
class MLPipelineInterface:
    def prepare_training_data(self, suppliers: List[Supplier], risk_events: List[RiskEvent]) -> Tuple[pd.DataFrame, ...]
    def train_models(self, X: pd.DataFrame, y_risk: pd.Series, y_impact: pd.Series, y_probability: pd.Series) -> Dict[str, float]
    def predict_risks(self, suppliers: List[Supplier], scraped_data: List[Dict]) -> List[MLPrediction]
```

**Model Architecture:**
- `RandomForestClassifier` for risk level classification (low, medium, high, critical)
- `GradientBoostingRegressor` for impact score prediction (0-10 scale)
- `RandomForestRegressor` for disruption probability prediction (0-1 scale)
- Feature engineering pipeline with financial, geographic, and content-based features

#### 3. AI Agent Orchestration
**Purpose:** Multi-agent analysis and strategic recommendation generation

**CrewAI Agent Definitions:**
```python
# Data Analysis Agent
data_analysis_agent = Agent(
    role='Supply Chain Data Analyst',
    goal='Analyze collected data and identify patterns, anomalies, and risk indicators',
    backstory='Expert analyst with deep knowledge of supply chain operations and risk factors.',
    allow_delegation=False,  # Specialist focused on data analysis
    verbose=True
)

# Risk Assessment Agent  
risk_assessment_agent = Agent(
    role='Risk Assessment Expert',
    goal='Evaluate and categorize risks based on data analysis and ML predictions',
    backstory='Senior risk manager with experience in global supply chain risk assessment.',
    allow_delegation=True,  # Can coordinate with other agents
    verbose=True
)

# Strategy Agent
strategy_agent = Agent(
    role='Strategic Advisor',
    goal='Generate actionable recommendations and mitigation strategies',
    backstory='Strategic consultant specializing in supply chain optimization and risk mitigation.',
    allow_delegation=False,  # Specialist focused on strategy
    verbose=True
)
```

**Task Coordination:**
- Sequential process execution with task dependencies using `context` parameter
- Collaborative analysis through agent delegation and information sharing
- Structured output validation using Pydantic models

#### 4. Workflow Orchestration Engine
**Purpose:** LangGraph-based workflow management with state persistence and error handling

**LangGraph Workflow Design:**
```python
class CompleteWorkflowState(TypedDict):
    # Input and configuration
    suppliers: List[Dict[str, Any]]
    analysis_config: Dict[str, Any]
    
    # State tracking
    current_step: str
    error_message: Optional[str]
    retry_count: int
    workflow_id: str
    
    # Processing results
    scraped_data: List[Dict[str, Any]]
    ml_predictions: List[Dict[str, Any]]
    crew_analysis: Dict[str, Any]
    validated_results: Optional[AnalysisResult]
```

**Workflow Nodes:**
- `initialize_workflow`: Setup and validation
- `collect_web_data`: Crawl4AI-based data collection
- `collect_api_data`: External API integration
- `train_ml_models`: Model training and validation
- `generate_ml_predictions`: Risk prediction generation
- `run_crew_analysis`: Multi-agent analysis execution
- `validate_and_structure`: Pydantic-based result validation
- `generate_final_report`: Comprehensive report generation

**Error Handling and Recovery:**
- Conditional routing based on processing results
- Retry mechanisms with exponential backoff
- Checkpoint-based recovery using `SqliteSaver`
- Graceful degradation with partial results

#### 5. Visualization and Reporting Engine
**Purpose:** Interactive dashboard and comprehensive reporting capabilities

**Streamlit Dashboard Components:**
- Executive summary metrics with real-time updates
- Interactive risk heatmaps using Plotly and Seaborn
- ML model performance dashboards
- Detailed analysis results with tabbed interfaces
- Export functionality for JSON, CSV, and PDF formats

**Visualization Features:**
- Risk correlation matrices and geographic distribution maps
- Time series analysis for trend identification
- Feature importance visualizations for ML model interpretability
- Agent performance metrics and collaboration tracking

### Data Models

#### Core Pydantic Models

```python
class Supplier(BaseModel):
    id: str = Field(..., description="Unique supplier identifier")
    name: str = Field(..., min_length=1, max_length=200)
    website: Optional[str] = Field(None, description="Company website for scraping")
    country: str = Field(..., description="Primary operating country")
    region: str = Field(..., description="Geographic region")
    industry: str = Field(..., description="Primary industry sector")
    annual_revenue: Optional[float] = Field(None, ge=0)
    employee_count: Optional[int] = Field(None, ge=1)
    financial_health_score: Optional[float] = Field(None, ge=0, le=100)
    tier: Literal["tier_1", "tier_2", "tier_3"] = Field(default="tier_2")
    criticality_score: float = Field(default=50.0, ge=0, le=100)

class RiskEvent(BaseModel):
    event_id: str
    supplier_id: str
    event_type: Literal["geopolitical", "financial", "environmental", "operational", "regulatory", "cyber"]
    severity: RiskLevel
    probability: float = Field(..., ge=0, le=1, description="ML predicted probability")
    impact_score: float = Field(..., ge=0, le=10, description="Business impact score")
    confidence_level: float = Field(..., ge=0, le=1, description="Model confidence")
    description: str = Field(..., min_length=10)
    evidence: List[str] = Field(default_factory=list)
    data_sources: List[DataSource] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.now)
    mitigation_actions: List[str] = Field(default_factory=list)

class AnalysisResult(BaseModel):
    analysis_id: str
    suppliers: List[Supplier]
    scraped_data: List[ScrapedData] = Field(default_factory=list)
    risk_events: List[RiskEvent] = Field(default_factory=list)
    ml_predictions: List[MLPrediction] = Field(default_factory=list)
    overall_risk_score: float = Field(..., ge=0, le=100)
    recommendations: List[str] = Field(default_factory=list)
    data_quality_score: float = Field(default=0.8, ge=0, le=1)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)
```

## Error Handling

### Comprehensive Error Management Strategy

#### 1. Data Collection Error Handling
**Web Scraping Resilience:**
- Multiple extraction strategies with fallback mechanisms
- Retry logic with exponential backoff for failed requests
- Content validation and quality scoring
- Graceful handling of rate limits and blocked requests

```python
async def robust_extraction():
    strategies = [
        RegexExtractionStrategy(pattern=RegexExtractionStrategy.Currency),
        JsonCssExtractionStrategy(product_schema),
        LLMExtractionStrategy(llm_config=fallback_config)
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            result = await crawler.arun(url=url, config=CrawlerRunConfig(extraction_strategy=strategy))
            if result.success and result.extracted_content:
                return json.loads(result.extracted_content)
        except Exception as e:
            logger.warning(f"Strategy {i+1} failed: {e}")
            continue
    
    raise ExtractionFailedException("All extraction strategies failed")
```

#### 2. ML Pipeline Error Handling
**Model Training and Prediction Resilience:**
- Data validation before model training
- Model performance monitoring and alerts
- Fallback to simpler models if complex models fail
- Confidence thresholds for prediction reliability

```python
def train_models_with_validation(self, X, y_risk, y_impact, y_probability):
    try:
        # Validate input data
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        # Train with cross-validation
        cv_scores = cross_val_score(self.risk_classifier, X, y_risk, cv=5)
        if cv_scores.mean() < 0.6:
            logger.warning("Model performance below threshold, using simpler model")
            self.risk_classifier = LogisticRegression()
        
        self.risk_classifier.fit(X, y_risk)
        return {"accuracy": cv_scores.mean()}
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise ModelTrainingException(f"Failed to train models: {str(e)}")
```

#### 3. Workflow Error Handling
**LangGraph State Management:**
- Checkpoint-based recovery for long-running workflows
- Conditional routing to error handling nodes
- Partial result preservation and graceful degradation
- Detailed error logging and user notification

```python
def _data_collection_router(self, state: CompleteWorkflowState) -> str:
    status = state["data_collection_status"]
    retry_count = state["retry_count"]
    
    if status == "web_complete":
        return "success"
    elif retry_count < 3 and status in ["web_error", "api_error"]:
        return "retry"
    else:
        return "error"

async def _handle_error(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
    logger.error(f"Workflow error in step '{state['current_step']}': {state.get('error_message')}")
    
    # Preserve partial results
    state.update({
        "analysis_complete": False,
        "final_report": {
            "error": state.get("error_message"),
            "failed_step": state["current_step"],
            "partial_results": self._extract_partial_results(state)
        }
    })
    
    return state
```

#### 4. Agent Coordination Error Handling
**CrewAI Task Management:**
- Task timeout handling and agent recovery
- Delegation failure management
- Result validation and quality checks
- Agent performance monitoring

```python
def create_resilient_crew():
    return Crew(
        agents=[data_analyst, risk_assessor, strategist],
        tasks=[analysis_task, assessment_task, strategy_task],
        process=Process.sequential,
        verbose=True,
        max_execution_time=1800,  # 30 minutes timeout
        step_callback=log_agent_performance
    )
```

## Testing Strategy

### Comprehensive Testing Framework

#### 1. Unit Testing
**Component-Level Testing:**
- Pydantic model validation testing
- ML pipeline component testing
- Data collection function testing
- Agent behavior verification

```python
class TestSupplyChainMLPipeline:
    def test_prepare_training_data(self):
        pipeline = SupplyChainMLPipeline()
        suppliers = [create_test_supplier()]
        risk_events = [create_test_risk_event()]
        
        X, y_risk, y_impact, y_probability = pipeline.prepare_training_data(suppliers, risk_events, [])
        
        assert X.shape[0] == len(suppliers)
        assert len(y_risk) == len(suppliers)
        assert all(col in X.columns for col in pipeline.feature_names)
    
    def test_model_training_validation(self):
        pipeline = SupplyChainMLPipeline()
        X, y_risk, y_impact, y_probability = create_test_training_data()
        
        performance = pipeline.train_models(X, y_risk, y_impact, y_probability)
        
        assert performance['risk_accuracy'] > 0.5
        assert performance['impact_r2'] > 0.3
        assert pipeline.is_trained == True
```

#### 2. Integration Testing
**Cross-Component Testing:**
- End-to-end workflow testing
- Agent collaboration testing
- Data flow validation
- Error propagation testing

```python
class TestWorkflowIntegration:
    async def test_complete_workflow_execution(self):
        workflow = CompleteSupplyChainWorkflow(test_api_key)
        suppliers = create_test_suppliers()
        
        result = await workflow.run_complete_analysis(suppliers)
        
        assert result["analysis_complete"] == True
        assert "final_report" in result
        assert len(result["final_report"]["detailed_results"]["suppliers"]) > 0
    
    async def test_error_recovery(self):
        workflow = CompleteSupplyChainWorkflow(invalid_api_key)
        suppliers = create_test_suppliers()
        
        result = await workflow.run_complete_analysis(suppliers)
        
        assert result["analysis_complete"] == False
        assert "error" in result
        assert "partial_results" in result.get("final_report", {})
```

#### 3. Performance Testing
**Scalability and Load Testing:**
- Concurrent workflow execution testing
- Large dataset processing validation
- Memory usage monitoring
- Response time benchmarking

```python
class TestPerformance:
    async def test_concurrent_workflows(self):
        workflows = [CompleteSupplyChainWorkflow(api_key) for _ in range(5)]
        suppliers_list = [create_test_suppliers(10) for _ in range(5)]
        
        start_time = time.time()
        results = await asyncio.gather(*[
            workflow.run_complete_analysis(suppliers) 
            for workflow, suppliers in zip(workflows, suppliers_list)
        ])
        execution_time = time.time() - start_time
        
        assert all(result["analysis_complete"] for result in results)
        assert execution_time < 300  # Should complete within 5 minutes
    
    def test_memory_usage(self):
        initial_memory = psutil.Process().memory_info().rss
        
        # Process large dataset
        pipeline = SupplyChainMLPipeline()
        large_dataset = create_large_test_dataset(1000)
        pipeline.train_models(*large_dataset)
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 500 * 1024 * 1024  # Less than 500MB increase
```

#### 4. Data Quality Testing
**Validation and Consistency Testing:**
- Scraped data quality validation
- ML prediction consistency testing
- Pydantic model constraint verification
- Data lineage tracking

```python
class TestDataQuality:
    def test_scraped_data_validation(self):
        collector = AdvancedDataCollector(api_key)
        scraped_data = create_test_scraped_data()
        
        for data in scraped_data:
            validated_data = ScrapedData(**data)
            assert validated_data.relevance_score >= 0 and validated_data.relevance_score <= 1
            assert validated_data.scraped_at <= datetime.now()
            assert len(validated_data.content) > 0
    
    def test_ml_prediction_consistency(self):
        pipeline = SupplyChainMLPipeline()
        pipeline.load_models("test_models/")
        
        suppliers = create_test_suppliers()
        scraped_data = create_test_scraped_data()
        
        # Run predictions multiple times
        predictions_1 = pipeline.predict_risks(suppliers, scraped_data)
        predictions_2 = pipeline.predict_risks(suppliers, scraped_data)
        
        # Results should be identical for same input
        assert len(predictions_1) == len(predictions_2)
        for p1, p2 in zip(predictions_1, predictions_2):
            assert p1.prediction == p2.prediction
            assert p1.confidence == p2.confidence
```

This comprehensive design provides a robust foundation for implementing the Supply Chain Intelligence & Risk Management Platform, ensuring scalability, reliability, and maintainability while leveraging the full capabilities of LangGraph, CrewAI, Crawl4AI, and other integrated technologies.