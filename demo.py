"""
Demonstration Script for Hierarchical Requirements Classification
================================================================

This script demonstrates the complete AI workflow for requirements engineering.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from hierarchical_workflow import HierarchicalRequirementsWorkflow
from benchmark_evaluation import BenchmarkEvaluator
from nfr_subclass_classifier import get_sample_nfr_data


def run_demo():
    """Run a complete demonstration of the hierarchical workflow"""
    print("HIERARCHICAL REQUIREMENTS CLASSIFICATION DEMO")
    print("=" * 55)
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("WARNING: Please set ANTHROPIC_API_KEY environment variable")
        print("You can do this by:")
        print("1. Copy .env.template to .env")
        print("2. Edit .env and add your Anthropic API key")
        print("3. Run: source .env (Linux/Mac) or set variables manually")
        return False
    
    try:
        # Initialize workflow
        print("\nInitializing AI workflow...")
        workflow = HierarchicalRequirementsWorkflow(
            anthropic_api_key=api_key,
            use_few_shot=True
        )
        print("SUCCESS: Workflow initialized successfully")
        
        # Get sample data
        print("\nLoading sample requirements...")
        requirements, true_labels = get_sample_nfr_data()
        print(f"SUCCESS: Loaded {len(requirements)} sample requirements")
        
        # Test single classification
        print("\nTesting single requirement classification...")
        sample_req = requirements[0]
        result = workflow.classify_single_requirement(sample_req)
        
        print(f"Requirement: {sample_req}")
        print(f"FR/NFR Classification: {result.fr_nfr_prediction} (conf: {result.fr_nfr_confidence:.2f})")
        if result.nfr_subclass_prediction:
            print(f"NFR Subclass: {result.nfr_subclass_prediction} (conf: {result.nfr_subclass_confidence:.2f})")
        print(f"Final Classification: {result.final_classification}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
        # Test batch classification
        print("\nTesting batch classification (first 3 requirements)...")
        batch_results = workflow.classify_batch(
            requirements[:3], 
            batch_size=3,
            delay_between_batches=1.0
        )
        
        for i, result in enumerate(batch_results):
            print(f"\n#{i+1}: {result.requirement_text[:60]}...")
            print(f"  -> {result.final_classification}")
        
        # Save results
        workflow.save_workflow_results(batch_results, experiment_name="demo_run")
        print("\nSUCCESS: Demo results saved to hierarchical_evaluation_results/")
        
        print("\nDEMO COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run full evaluation: python benchmark_evaluation.py")
        print("2. Explore results in evaluation_results/ directories")
        print("3. Customize prompts and parameters for your use case")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = run_demo()
    sys.exit(0 if success else 1)
