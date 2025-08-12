# CrewAI Multi-Agent System for Supply Chain Intelligence

This directory contains the complete implementation of the CrewAI multi-agent system for supply chain intelligence and risk management.

## 🎯 Requirements Covered

- **3.1**: Data Analysis Agent with pattern recognition capabilities
- **3.2**: ML Specialist Agent for model validation and interpretation  
- **3.3**: Risk Assessment Agent for risk categorization and prioritization
- **3.4**: Strategy Agent for recommendation generation
- **3.5**: Agent coordination and task management with error handling
- **3.6**: Agent task definitions with structured outputs

## 🏗️ Architecture

### Specialized Agents (`specialized_agents.py`)

Four specialized AI agents with distinct roles:

1. **Data Analysis Agent**
   - Identifies patterns, anomalies, and risk indicators
   - Analyzes supplier data and web-scraped content
   - Provides data quality assessments

2. **ML Specialist Agent**
   - Validates machine learning model predictions
   - Assesses confidence levels and model reliability
   - Interprets feature importance and model performance

3. **Risk Assessment Agent**
   - Categorizes risks by type (geopolitical, financial, environmental, etc.)
   - Evaluates business impact and likelihood timelines
   - Creates priority matrices for risk management

4. **Strategy Agent**
   - Generates actionable recommendations
   - Develops mitigation strategies and implementation roadmaps
   - Provides cost-benefit analysis and success metrics

### Coordination System (`coordination.py`)

- **AgentCoordinator**: Manages agent collaboration and task execution
- **Error Handling**: Retry logic with exponential backoff
- **Performance Monitoring**: Tracks execution metrics and agent performance
- **Context Sharing**: Enables sequential task execution with shared context

### Task Definitions (`task_definitions.py`)

Structured tasks with Pydantic output models:

- **DataAnalysisOutput**: Patterns, anomalies, risk indicators
- **MLValidationOutput**: Model performance and confidence assessment
- **RiskAssessmentOutput**: Risk categorization and priority matrix
- **StrategyOutput**: Recommendations and implementation roadmap

### Orchestrator (`crew_orchestrator.py`)

Main interface for running complete analysis workflows:

- **Complete Analysis**: End-to-end supply chain intelligence analysis
- **Individual Analysis**: Run specific agent tasks independently
- **Result Management**: Structured output processing and file saving
- **Performance Tracking**: Execution history and metrics

## 🚀 Usage

### Quick Start

```python
from agents import run_supply_chain_analysis

# Run complete analysis
result = run_supply_chain_analysis(
    suppliers_data=suppliers,
    scraped_data=scraped_content,
    ml_predictions=predictions,
    model_metadata=model_info,
    business_context=context
)
```

### Advanced Usage

```python
from agents import SupplyChainCrewOrchestrator

# Initialize orchestrator
orchestrator = SupplyChainCrewOrchestrator(
    llm_model="gpt-4o",
    verbose=True,
    enable_memory=True
)

# Run complete analysis
result = orchestrator.run_complete_analysis(
    suppliers_data=suppliers,
    scraped_data=scraped_content,
    ml_predictions=predictions,
    model_metadata=model_info,
    business_context=context,
    save_results=True
)

# Run individual agent analysis
data_result = orchestrator.run_individual_analysis(
    analysis_type="data",
    suppliers_data=suppliers,
    scraped_data=scraped_content
)
```

### Example Script

Run the example script to see the system in action:

```bash
python -m agents.example_usage
```

## 📊 Output Structure

The system provides structured outputs using Pydantic models:

### Data Analysis Output
- Identified patterns and anomalies
- Risk indicators with confidence scores
- Data quality assessments
- Key insights and findings

### ML Validation Output
- Model performance metrics
- Prediction confidence assessments
- Feature importance rankings
- Model reliability classification

### Risk Assessment Output
- Risk categorization by type and severity
- Business impact assessments
- Likelihood timelines and priority matrix
- Critical risks requiring immediate attention

### Strategy Output
- Actionable recommendations with implementation details
- Risk mitigation strategies and timelines
- Cost-benefit analysis and success metrics
- Executive summary for C-level presentation

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Agent Configuration

Agents can be configured with different LLM models:

```python
orchestrator = SupplyChainCrewOrchestrator(
    llm_model="gpt-4o",  # or "claude-3-5-sonnet", "gemini-pro", etc.
    verbose=True,
    enable_memory=True,
    max_retries=3
)
```

## 📈 Performance Monitoring

The system includes comprehensive performance monitoring:

- Execution time tracking
- Success/failure rates
- Agent-specific performance metrics
- Error logging and retry statistics

Access performance metrics:

```python
# Get performance summary
summary = orchestrator.get_performance_summary()

# Get execution history
history = orchestrator.get_execution_history()

# Save metrics to file
orchestrator.coordinator.save_performance_metrics("metrics.json")
```

## 🛠️ Error Handling

Robust error handling includes:

- Automatic retry with exponential backoff
- Graceful degradation with partial results
- Detailed error logging and reporting
- Task dependency management

## 🔄 Integration

The multi-agent system integrates with other platform components:

- **Data Collection**: Processes scraped data and supplier information
- **ML Pipeline**: Validates and interprets ML model predictions
- **Workflow Orchestration**: Can be integrated with LangGraph workflows
- **Visualization**: Outputs can be consumed by Streamlit dashboard

## 📝 Files Overview

- `specialized_agents.py`: Four specialized AI agents
- `coordination.py`: Agent coordination and task management
- `task_definitions.py`: Structured task definitions with Pydantic models
- `crew_orchestrator.py`: Main orchestrator for complete workflows
- `example_usage.py`: Example scripts and usage demonstrations
- `__init__.py`: Package exports and public API

## 🎯 Next Steps

The CrewAI multi-agent system is now ready for integration with:

1. **LangGraph Workflows**: For end-to-end orchestration
2. **Streamlit Dashboard**: For interactive visualization
3. **ML Pipeline**: For real-time prediction validation
4. **Data Collection**: For automated analysis triggers

The system provides a solid foundation for collaborative AI-powered supply chain intelligence and risk management.