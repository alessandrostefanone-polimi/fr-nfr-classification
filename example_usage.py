#!/usr/bin/env python3
"""
Example usage of the FR/NFR Classification Pipeline

This script demonstrates how to use the classification pipeline
with different LLM configurations.
"""

import os
from langchain_fr_nfr_pipeline import RequirementClassifier, DatasetLoader

def main():
    """Example usage of the classification pipeline"""
    
    # Set your API key (or use .env file)
    # os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    print("FR/NFR Classification Pipeline Example")
    print("=" * 50)
    
    # Example 1: Quick test with sample data
    print("\n1. Testing with sample requirements...")
    
    # Initialize classifier
    classifier = RequirementClassifier(
        model_type="claude",
        model_name="claude-3-5-haiku-20241022",
        use_few_shot=True,
        temperature=0.1
    )
    
    # Test requirements
    test_requirements = [
        "The system shall calculate the monthly interest rate based on the principal amount.",
        "The system shall respond to user queries within 2 seconds.",
        "Users shall be able to create new customer records in the database.",
        "The application must maintain 99.9% uptime during business hours."
    ]
    
    # Classify each requirement
    for i, req in enumerate(test_requirements, 1):
        print(f"\n--- Requirement {i} ---")
        print(f"Text: {req}")
        
        result = classifier.classify_requirement(req)
        print(f"Classification: {result.predicted_label}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Key Indicators: {', '.join(result.key_indicators)}")
    
    # Example 2: Batch classification with evaluation
    print("\n\n2. Batch classification with evaluation...")
    
    # Load test data
    test_reqs, test_labels = DatasetLoader.create_test_samples()
    
    # Classify batch
    results = classifier.classify_batch(test_reqs, test_labels)
    
    # Evaluate performance
    metrics = classifier.evaluate_performance(results)
    
    print(f"\nPerformance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    
    # Save results
    classifier.save_results(results, metrics, "output/example_results/")
    print("\nResults saved to output/example_results/")

if __name__ == "__main__":
    main()
