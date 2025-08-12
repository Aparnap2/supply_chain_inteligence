"""
Example Usage of Complete Supply Chain Workflow

This module demonstrates how to use the complete workflow orchestration
for supply chain intelligence analysis.
"""

import asyncio
import logging
from typing import List

from .complete_workflow import CompleteSupplyChainWorkflow
from models.supplier import Supplier


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_example_analysis():
    """
    Run an example supply chain analysis workflow.
    """
    # Create sample suppliers
    suppliers = [
        Supplier(
            id="SUPP_001",
            name="TechCorp Manufacturing",
            website="https://techcorp.example.com",
            country="China",
            region="Asia",
            industry="Manufacturing",
            annual_revenue=500000000.0,
            employee_count=2500,
            financial_health_score=75.0,
            tier="tier_1",
            criticality_score=85.0
        ),
        Supplier(
            id="SUPP_002", 
            name="Global Components Ltd",
            website="https://globalcomponents.example.com",
            country="Germany",
            region="Europe",
            industry="Manufacturing",
            annual_revenue=200000000.0,
            employee_count=800,
            financial_health_score=82.0,
            tier="tier_2",
            criticality_score=70.0
        ),
        Supplier(
            id="SUPP_003",
            name="Innovation Materials Inc",
            website="https://innovationmaterials.example.com",
            country="United States",
            region="North America", 
            industry="Materials",
            annual_revenue=150000000.0,
            employee_count=600,
            financial_health_score=68.0,
            tier="tier_2",
            criticality_score=60.0
        )
    ]
    
    # Configuration for analysis
    analysis_config = {
        "enable_web_scraping": True,
        "enable_api_collection": True,
        "enable_ml_training": True,
        "enable_crew_analysis": True,
        "max_retries": 3
    }
    
    # Initialize workflow (you would provide your OpenAI API key here)
    workflow = CompleteSupplyChainWorkflow(
        api_key=None,  # Set to your OpenAI API key for full functionality
        checkpoint_db_path="checkpoints/example_workflow.db"
    )
    
    logger.info("Starting example supply chain analysis...")
    
    try:
        # Run complete analysis
        result = await workflow.run_complete_analysis(
            suppliers=suppliers,
            analysis_config=analysis_config
        )
        
        # Display results
        print("\n" + "="*60)
        print("SUPPLY CHAIN ANALYSIS RESULTS")
        print("="*60)
        
        print(f"Workflow ID: {result['workflow_id']}")
        print(f"Success: {result['success']}")
        
        if result['success']:
            final_report = result['final_report']
            
            print(f"\nExecution Summary:")
            print(f"  Total Duration: {final_report['execution_summary']['total_duration_ms']:.2f}ms")
            print(f"  Steps Completed: {len(final_report['execution_summary']['steps_completed'])}")
            print(f"  Steps Failed: {len(final_report['execution_summary']['steps_failed'])}")
            
            print(f"\nData Summary:")
            print(f"  Suppliers Analyzed: {final_report['data_summary']['suppliers_analyzed']}")
            print(f"  Web Data Collected: {final_report['data_summary']['web_data_collected']}")
            print(f"  API Data Sources: {final_report['data_summary']['api_data_sources']}")
            print(f"  ML Predictions: {final_report['data_summary']['ml_predictions_generated']}")
            print(f"  Data Quality: {final_report['data_summary']['overall_data_quality']:.2f}")
            
            print(f"\nRisk Assessment:")
            print(f"  Overall Risk Score: {final_report['risk_assessment']['overall_risk_score']:.2f}")
            print(f"  High Risk Suppliers: {len(final_report['risk_assessment']['high_risk_suppliers'])}")
            
            if final_report['risk_assessment']['high_risk_suppliers']:
                print(f"  High Risk Supplier IDs: {', '.join(final_report['risk_assessment']['high_risk_suppliers'])}")
            
            print(f"\nRecommendations:")
            for i, rec in enumerate(final_report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        else:
            print(f"Analysis failed: {result.get('error', 'Unknown error')}")
            
            if 'execution_metadata' in result:
                metadata = result['execution_metadata']
                print(f"Failed steps: {metadata.get('steps_failed', [])}")
                print(f"Retry count: {metadata.get('retry_count', 0)}")
        
        print("\n" + "="*60)
        
        return result
        
    except Exception as e:
        logger.error(f"Example analysis failed: {str(e)}")
        raise


async def demonstrate_workflow_recovery():
    """
    Demonstrate workflow recovery from checkpoint.
    """
    logger.info("Demonstrating workflow recovery...")
    
    # Initialize workflow
    workflow = CompleteSupplyChainWorkflow(
        api_key=None,
        checkpoint_db_path="checkpoints/recovery_example.db"
    )
    
    # Create a simple supplier for testing
    suppliers = [
        Supplier(
            id="SUPP_RECOVERY",
            name="Recovery Test Supplier",
            country="United States",
            region="North America",
            industry="Technology"
        )
    ]
    
    try:
        # Start analysis
        result = await workflow.run_complete_analysis(suppliers)
        workflow_id = result['workflow_id']
        
        print(f"\nOriginal workflow {workflow_id} completed: {result['success']}")
        
        # Demonstrate status checking
        status = workflow.get_workflow_status(workflow_id)
        print(f"Workflow status: {status['status']}")
        print(f"Current step: {status['current_step']}")
        print(f"Progress: {status['progress']}")
        
        # Demonstrate resuming (would work if workflow was interrupted)
        resume_result = await workflow.resume_workflow(workflow_id)
        print(f"Resume attempt: {resume_result['success']}")
        
    except Exception as e:
        logger.error(f"Recovery demonstration failed: {str(e)}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_example_analysis())
    
    # Uncomment to also run recovery demonstration
    # asyncio.run(demonstrate_workflow_recovery())