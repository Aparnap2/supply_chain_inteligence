"""
Example Usage of CrewAI Multi-Agent System

This script demonstrates how to use the CrewAI multi-agent system for
supply chain intelligence analysis.
"""

import os
from typing import Dict, List, Any
from datetime import datetime

from .crew_orchestrator import SupplyChainCrewOrchestrator


def create_sample_data() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Create sample data for testing the CrewAI multi-agent system
    
    Returns:
        Tuple of (suppliers_data, scraped_data, ml_predictions, model_metadata, business_context)
    """
    
    # Sample supplier data
    suppliers_data = [
        {
            "id": "SUP001",
            "name": "TechComponents Inc",
            "website": "https://techcomponents.com",
            "country": "China",
            "region": "Asia Pacific",
            "industry": "Electronics Manufacturing",
            "annual_revenue": 50000000,
            "employee_count": 500,
            "financial_health_score": 75.5,
            "tier": "tier_1",
            "criticality_score": 85.0
        },
        {
            "id": "SUP002", 
            "name": "Global Logistics Solutions",
            "website": "https://globallogistics.com",
            "country": "Germany",
            "region": "Europe",
            "industry": "Logistics & Transportation",
            "annual_revenue": 120000000,
            "employee_count": 1200,
            "financial_health_score": 82.3,
            "tier": "tier_1",
            "criticality_score": 90.0
        },
        {
            "id": "SUP003",
            "name": "Raw Materials Corp",
            "website": "https://rawmaterials.com", 
            "country": "Brazil",
            "region": "South America",
            "industry": "Raw Materials",
            "annual_revenue": 25000000,
            "employee_count": 200,
            "financial_health_score": 68.7,
            "tier": "tier_2",
            "criticality_score": 70.0
        }
    ]
    
    # Sample scraped data
    scraped_data = [
        {
            "url": "https://techcomponents.com/news",
            "content": "TechComponents Inc announces expansion of manufacturing capacity by 30% to meet growing demand for semiconductor components.",
            "scraped_at": datetime.now().isoformat(),
            "relevance_score": 0.85,
            "sentiment_score": 0.7,
            "data_source": "company_website",
            "supplier_id": "SUP001"
        },
        {
            "url": "https://news.industry.com/logistics-disruption",
            "content": "European logistics companies face potential disruptions due to new environmental regulations affecting transportation routes.",
            "scraped_at": datetime.now().isoformat(),
            "relevance_score": 0.92,
            "sentiment_score": -0.3,
            "data_source": "industry_news",
            "supplier_id": "SUP002"
        },
        {
            "url": "https://economic-times.com/brazil-commodity-prices",
            "content": "Brazil commodity prices show volatility amid global economic uncertainty, affecting raw material suppliers in the region.",
            "scraped_at": datetime.now().isoformat(),
            "relevance_score": 0.78,
            "sentiment_score": -0.5,
            "data_source": "economic_news",
            "supplier_id": "SUP003"
        }
    ]
    
    # Sample ML predictions
    ml_predictions = [
        {
            "supplier_id": "SUP001",
            "risk_level": "medium",
            "risk_probability": 0.35,
            "impact_score": 6.2,
            "confidence": 0.82,
            "predicted_at": datetime.now().isoformat(),
            "model_version": "v1.2.0"
        },
        {
            "supplier_id": "SUP002", 
            "risk_level": "high",
            "risk_probability": 0.65,
            "impact_score": 8.1,
            "confidence": 0.78,
            "predicted_at": datetime.now().isoformat(),
            "model_version": "v1.2.0"
        },
        {
            "supplier_id": "SUP003",
            "risk_level": "medium",
            "risk_probability": 0.42,
            "impact_score": 5.8,
            "confidence": 0.71,
            "predicted_at": datetime.now().isoformat(),
            "model_version": "v1.2.0"
        }
    ]
    
    # Sample model metadata
    model_metadata = {
        "risk_classifier": {
            "model_type": "RandomForestClassifier",
            "accuracy": 0.84,
            "precision": 0.82,
            "recall": 0.86,
            "f1_score": 0.84,
            "training_date": "2024-01-15",
            "feature_count": 25
        },
        "impact_regressor": {
            "model_type": "GradientBoostingRegressor",
            "r2_score": 0.73,
            "mae": 0.85,
            "rmse": 1.12,
            "training_date": "2024-01-15",
            "feature_count": 25
        },
        "probability_regressor": {
            "model_type": "RandomForestRegressor",
            "r2_score": 0.69,
            "mae": 0.08,
            "rmse": 0.12,
            "training_date": "2024-01-15",
            "feature_count": 25
        }
    }
    
    # Sample business context
    business_context = {
        "company_name": "Global Manufacturing Corp",
        "industry": "Automotive Manufacturing",
        "annual_revenue": 2500000000,
        "risk_tolerance": "medium",
        "budget_constraints": {
            "risk_mitigation_budget": 5000000,
            "timeline": "12_months"
        },
        "strategic_priorities": [
            "supply_chain_resilience",
            "cost_optimization", 
            "sustainability",
            "digital_transformation"
        ],
        "compliance_requirements": [
            "ISO_27001",
            "SOX_compliance",
            "GDPR"
        ],
        "geographic_focus": ["North America", "Europe", "Asia Pacific"]
    }
    
    return suppliers_data, scraped_data, ml_predictions, model_metadata, business_context


def run_example_analysis():
    """
    Run an example analysis using the CrewAI multi-agent system
    """
    print("🚀 Starting CrewAI Multi-Agent Supply Chain Analysis Example")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key to run this example")
        return
    
    try:
        # Create sample data
        print("📊 Creating sample data...")
        suppliers_data, scraped_data, ml_predictions, model_metadata, business_context = create_sample_data()
        
        print(f"   - {len(suppliers_data)} suppliers")
        print(f"   - {len(scraped_data)} scraped data items")
        print(f"   - {len(ml_predictions)} ML predictions")
        
        # Initialize orchestrator
        print("\n🤖 Initializing CrewAI Multi-Agent System...")
        orchestrator = SupplyChainCrewOrchestrator(
            llm_model="gpt-4o",
            verbose=True,
            enable_memory=True
        )
        
        print(f"   - Agent roles: {', '.join(orchestrator.agent_factory.get_agent_roles())}")
        
        # Run complete analysis
        print("\n🔍 Running complete supply chain analysis...")
        print("This may take several minutes as agents collaborate...")
        
        result = orchestrator.run_complete_analysis(
            suppliers_data=suppliers_data,
            scraped_data=scraped_data,
            ml_predictions=ml_predictions,
            model_metadata=model_metadata,
            business_context=business_context,
            save_results=True,
            results_dir="example_results"
        )
        
        # Display results
        print("\n📋 Analysis Results:")
        print("=" * 40)
        
        if result["success"]:
            print(f"✅ Analysis completed successfully!")
            print(f"   - Execution ID: {result['execution_id']}")
            print(f"   - Execution time: {result['execution_time']:.2f} seconds")
            print(f"   - Results saved to: example_results/")
            
            # Show structured outputs summary
            if "structured_outputs" in result:
                print(f"\n📊 Structured Outputs Generated:")
                for task_name, output in result["structured_outputs"].items():
                    if isinstance(output, dict):
                        print(f"   - {task_name}: {len(output)} fields")
                    else:
                        print(f"   - {task_name}: {type(output).__name__}")
            
            # Show performance metrics
            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                print(f"\n⚡ Performance Metrics:")
                print(f"   - Total executions: {metrics.get('total_executions', 0)}")
                print(f"   - Success rate: {metrics.get('successful_executions', 0)}/{metrics.get('total_executions', 0)}")
                print(f"   - Average execution time: {metrics.get('average_execution_time', 0):.2f}s")
        
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            print(f"   - Execution ID: {result['execution_id']}")
            print(f"   - Execution time: {result.get('execution_time', 0):.2f} seconds")
        
        # Show execution history
        print(f"\n📈 Execution History:")
        history = orchestrator.get_execution_history()
        for exec_record in history[-3:]:  # Show last 3 executions
            status = "✅" if exec_record["success"] else "❌"
            print(f"   {status} {exec_record['execution_id']} - {exec_record['timestamp']}")
        
        print("\n🎉 Example completed!")
        
    except Exception as e:
        print(f"\n❌ Error running example: {str(e)}")
        print("Please check your configuration and try again.")


def run_individual_agent_example():
    """
    Run an example of individual agent analysis
    """
    print("🔬 Running Individual Agent Analysis Example")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Create sample data
        suppliers_data, scraped_data, ml_predictions, model_metadata, business_context = create_sample_data()
        
        # Initialize orchestrator
        orchestrator = SupplyChainCrewOrchestrator(verbose=True)
        
        # Run data analysis only
        print("\n📊 Running Data Analysis Agent...")
        data_result = orchestrator.run_individual_analysis(
            analysis_type="data",
            suppliers_data=suppliers_data,
            scraped_data=scraped_data
        )
        
        if data_result["success"]:
            print("✅ Data analysis completed successfully!")
            print(f"   - Execution time: {data_result.get('execution_time', 0):.2f}s")
        else:
            print(f"❌ Data analysis failed: {data_result.get('error')}")
        
        # Run ML validation only
        print("\n🤖 Running ML Specialist Agent...")
        ml_result = orchestrator.run_individual_analysis(
            analysis_type="ml",
            ml_predictions=ml_predictions,
            model_metadata=model_metadata
        )
        
        if ml_result["success"]:
            print("✅ ML validation completed successfully!")
            print(f"   - Execution time: {ml_result.get('execution_time', 0):.2f}s")
        else:
            print(f"❌ ML validation failed: {ml_result.get('error')}")
        
        print("\n🎉 Individual agent examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error running individual agent example: {str(e)}")


if __name__ == "__main__":
    print("CrewAI Multi-Agent System Examples")
    print("=" * 40)
    print("1. Complete Analysis Example")
    print("2. Individual Agent Example")
    
    choice = input("\nSelect example (1 or 2): ").strip()
    
    if choice == "1":
        run_example_analysis()
    elif choice == "2":
        run_individual_agent_example()
    else:
        print("Invalid choice. Running complete analysis example...")
        run_example_analysis()