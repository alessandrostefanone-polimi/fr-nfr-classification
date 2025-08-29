"""
Hierarchical Requirements Classification Workflow
================================================

This module integrates your existing FR/NFR classification with LLM-based NFR subclassification
to create a complete AI workflow for requirements engineering.

Workflow:
1. Input Requirement Text
2. FR/NFR Classification (using existing classifier from alessandrostefanone-polimi/fr-nfr-classification)
3. If NFR → LLM-based Subclassification (O, PE, SE, US)
4. Output: Final hierarchical classification

Author: Alessandro Stefanone
Affiliation: Politecnico di Milano - PhD in Mechanical Engineering
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import your existing FR/NFR classifier
# Note: This will need to be adapted based on your actual implementation
try:
    # Assuming your existing classifier is in the current directory or PYTHONPATH
    from langchain_fr_nfr_pipeline import RequirementClassifier
    FR_NFR_AVAILABLE = True
except ImportError:
    print("⚠️  Existing FR/NFR classifier not found. Will simulate for demonstration.")
    FR_NFR_AVAILABLE = False

# Import the NFR subclassification module we just created
from nfr_subclass_classifier import (
    LLMNFRSubclassifier, 
    NFRSubclassResult, 
    NFRSubclassificationEvaluator,
    load_subclass_dataset,
    get_sample_nfr_data
)

# Dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️  Datasets library not available. Using sample data.")
    DATASETS_AVAILABLE = False


@dataclass
class HierarchicalResult:
    """Complete result structure for hierarchical classification"""
    requirement_text: str
    fr_nfr_prediction: str
    fr_nfr_confidence: float
    nfr_subclass_prediction: Optional[str] = None
    nfr_subclass_confidence: Optional[float] = None
    final_classification: str = None
    reasoning: str = ""
    processing_time: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        """Set final classification after initialization"""
        if self.fr_nfr_prediction == "NFR" and self.nfr_subclass_prediction:
            self.final_classification = f"NFR_{self.nfr_subclass_prediction}"
        else:
            self.final_classification = self.fr_nfr_prediction


class MockFRNFRClassifier:
    """Mock FR/NFR classifier for demonstration when existing classifier unavailable"""
    
    def __init__(self):
        self.model_name = "mock_fr_nfr_classifier"
    
    def classify_requirement(self, requirement_text: str) -> Dict:
        """Mock classification - assumes all inputs are NFRs for subclass testing"""
        time.sleep(0.1)  # Simulate processing time
        
        # Simple heuristic for demo purposes
        fr_indicators = ['shall provide', 'shall display', 'shall calculate', 'shall store', 'function']
        is_functional = any(indicator in requirement_text.lower() for indicator in fr_indicators)
        
        return {
            'predicted_label': 'FR' if is_functional else 'NFR',
            'confidence': 0.85,
            'reasoning': 'Mock classification based on simple heuristics'
        }


class HierarchicalRequirementsWorkflow:
    """
    Complete AI workflow for hierarchical requirements classification
    
    This integrates:
    1. Your existing FR/NFR classifier (from alessandrostefanone-polimi/fr-nfr-classification)
    2. New LLM-based NFR subclassifier
    """
    
    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        use_few_shot: bool = True,
        temperature: float = 0.0,
        fr_nfr_model_type: str = "claude"
    ):
        """
        Initialize the hierarchical workflow
        
        Args:
            anthropic_api_key: API key for Anthropic Claude
            use_few_shot: Whether to use few-shot learning for subclassification
            temperature: Temperature for LLM generation
            fr_nfr_model_type: Model type for FR/NFR classification
        """
        
        # Setup API key
        if anthropic_api_key:
            os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
        
        # Initialize FR/NFR classifier
        if FR_NFR_AVAILABLE:
            try:
                self.fr_nfr_classifier = RequirementClassifier(
                    model_type=fr_nfr_model_type,
                    use_few_shot=use_few_shot
                )
                print("✅ Loaded existing FR/NFR classifier")
            except Exception as e:
                print(f"⚠️  Error loading FR/NFR classifier: {e}")
                self.fr_nfr_classifier = MockFRNFRClassifier()
                print("🔄 Using mock FR/NFR classifier")
        else:
            self.fr_nfr_classifier = MockFRNFRClassifier()
            print("🔄 Using mock FR/NFR classifier for demonstration")
        
        # Initialize NFR subclassifier
        self.nfr_subclassifier = LLMNFRSubclassifier(
            api_key=anthropic_api_key,
            use_few_shot=use_few_shot,
            temperature=temperature
        )
        
        # Initialize evaluator
        self.evaluator = NFRSubclassificationEvaluator(
            output_dir="hierarchical_evaluation_results"
        )
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the workflow"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'hierarchical_workflow.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def classify_single_requirement(self, requirement_text: str) -> HierarchicalResult:
        """
        Classify a single requirement through the hierarchical workflow
        
        Args:
            requirement_text: The requirement text to classify
            
        Returns:
            HierarchicalResult with complete classification details
        """
        start_time = time.time()
        
        self.logger.info(f"Starting hierarchical classification: '{requirement_text[:50]}...'")
        
        # Step 1: FR/NFR Classification
        fr_nfr_result = self.fr_nfr_classifier.classify_requirement(requirement_text)
        
        # Extract results (handle different result formats)
        if hasattr(fr_nfr_result, 'predicted_label'):
            fr_nfr_pred = fr_nfr_result.predicted_label
            fr_nfr_conf = getattr(fr_nfr_result, 'confidence', 0.5)
            fr_nfr_reasoning = getattr(fr_nfr_result, 'reasoning', '')
        elif isinstance(fr_nfr_result, dict):
            fr_nfr_pred = fr_nfr_result.get('predicted_label', 'UNKNOWN')
            fr_nfr_conf = fr_nfr_result.get('confidence', 0.5)
            fr_nfr_reasoning = fr_nfr_result.get('reasoning', '')
        else:
            fr_nfr_pred = str(fr_nfr_result)
            fr_nfr_conf = 0.5
            fr_nfr_reasoning = 'No reasoning available'
        
        # Initialize result
        result = HierarchicalResult(
            requirement_text=requirement_text,
            fr_nfr_prediction=fr_nfr_pred,
            fr_nfr_confidence=fr_nfr_conf,
            reasoning=f"FR/NFR: {fr_nfr_reasoning}"
        )
        
        # Step 2: NFR Subclassification (if needed)
        if fr_nfr_pred == "NFR":
            self.logger.info("Requirement classified as NFR, proceeding with subclassification...")
            
            nfr_subresult = self.nfr_subclassifier.classify_requirement(requirement_text)
            
            result.nfr_subclass_prediction = nfr_subresult.predicted_label
            result.nfr_subclass_confidence = nfr_subresult.confidence
            result.reasoning += f" | NFR Subclass: {nfr_subresult.reasoning}"
            result.metadata = {
                'nfr_key_indicators': nfr_subresult.key_indicators,
                'nfr_processing_time': nfr_subresult.processing_time
            }
        
        # Calculate total processing time
        result.processing_time = time.time() - start_time
        
        # Set final classification
        result.__post_init__()
        
        self.logger.info(f"Final classification: {result.final_classification}")
        return result
    
    def classify_batch(
        self, 
        requirements: List[str],
        true_labels: Optional[List[str]] = None,
        batch_size: int = 10,
        delay_between_batches: float = 1.0
    ) -> List[HierarchicalResult]:
        """
        Classify a batch of requirements through the hierarchical workflow
        
        Args:
            requirements: List of requirement texts
            true_labels: Optional true labels for evaluation
            batch_size: Batch size for rate limiting
            delay_between_batches: Delay between batches
            
        Returns:
            List of HierarchicalResult objects
        """
        results = []
        total_requirements = len(requirements)
        
        self.logger.info(f"Starting batch hierarchical classification of {total_requirements} requirements")
        
        for i, requirement in enumerate(requirements):
            result = self.classify_single_requirement(requirement)
            results.append(result)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{total_requirements} requirements")
            
            # Rate limiting
            if (i + 1) % batch_size == 0 and i + 1 < total_requirements:
                self.logger.info(f"Batch complete. Waiting {delay_between_batches}s...")
                time.sleep(delay_between_batches)
        
        self.logger.info(f"Batch hierarchical classification complete")
        return results
    
    def evaluate_hierarchical_performance(
        self,
        results: List[HierarchicalResult],
        true_labels: List[str],
        evaluation_type: str = "subclass_only"
    ) -> Dict:
        """
        Evaluate the performance of hierarchical classification
        
        Args:
            results: List of classification results
            true_labels: True labels for comparison
            evaluation_type: Type of evaluation ('subclass_only', 'hierarchical', 'fr_nfr_only')
            
        Returns:
            Evaluation metrics dictionary
        """
        if evaluation_type == "subclass_only":
            # Evaluate only NFR subclassification performance
            nfr_results = []
            nfr_true_labels = []
            
            for result, true_label in zip(results, true_labels):
                if result.fr_nfr_prediction == "NFR" and result.nfr_subclass_prediction:
                    # Create NFRSubclassResult for evaluation
                    nfr_result = NFRSubclassResult(
                        requirement_text=result.requirement_text,
                        predicted_label=result.nfr_subclass_prediction,
                        confidence=result.nfr_subclass_confidence or 0.0,
                        reasoning=result.reasoning,
                        key_indicators=result.metadata.get('nfr_key_indicators', []) if result.metadata else [],
                        processing_time=result.metadata.get('nfr_processing_time', 0.0) if result.metadata else 0.0
                    )
                    nfr_results.append(nfr_result)
                    nfr_true_labels.append(true_label)
            
            if nfr_results:
                return self.evaluator.evaluate_results(nfr_results, nfr_true_labels)
            else:
                return {"error": "No NFR results found for evaluation"}
        
        elif evaluation_type == "hierarchical":
            # Evaluate complete hierarchical classification
            predictions = [r.final_classification for r in results]
            
            # Calculate hierarchical accuracy
            correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
            accuracy = correct / len(predictions)
            
            return {
                "hierarchical_accuracy": accuracy,
                "total_predictions": len(predictions),
                "correct_predictions": correct,
                "predictions": predictions,
                "true_labels": true_labels
            }
        
        else:
            return {"error": f"Unknown evaluation type: {evaluation_type}"}
    
    def save_workflow_results(
        self,
        results: List[HierarchicalResult],
        evaluation_metrics: Optional[Dict] = None,
        experiment_name: str = "hierarchical_workflow"
    ):
        """
        Save complete workflow results
        
        Args:
            results: List of hierarchical results
            evaluation_metrics: Optional evaluation metrics
            experiment_name: Name for the experiment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("hierarchical_evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrame
        results_data = []
        for result in results:
            row = asdict(result)
            # Flatten metadata if present
            if result.metadata:
                for key, value in result.metadata.items():
                    if isinstance(value, list):
                        row[f"metadata_{key}"] = ', '.join(map(str, value))
                    else:
                        row[f"metadata_{key}"] = value
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save results
        results_file = output_dir / f"{experiment_name}_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        
        # Save evaluation metrics if provided
        if evaluation_metrics:
            metrics_file = output_dir / f"{experiment_name}_metrics_{timestamp}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_metrics, f, indent=2, ensure_ascii=False)
            print(f"📊 Metrics saved to: {metrics_file}")
        
        print(f"💾 Results saved to: {results_file}")
        
        # Generate summary report
        self._generate_summary_report(results, evaluation_metrics, experiment_name, timestamp)
    
    def _generate_summary_report(
        self,
        results: List[HierarchicalResult],
        evaluation_metrics: Optional[Dict],
        experiment_name: str,
        timestamp: str
    ):
        """Generate a summary report of the workflow results"""
        output_dir = Path("hierarchical_evaluation_results")
        report_file = output_dir / f"{experiment_name}_summary_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("HIERARCHICAL REQUIREMENTS CLASSIFICATION WORKFLOW\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Requirements Processed: {len(results)}\n\n")
            
            # Classification distribution
            fr_count = sum(1 for r in results if r.fr_nfr_prediction == 'FR')
            nfr_count = sum(1 for r in results if r.fr_nfr_prediction == 'NFR')
            
            f.write("FR/NFR DISTRIBUTION:\n")
            f.write(f"  Functional Requirements (FR): {fr_count}\n")
            f.write(f"  Non-Functional Requirements (NFR): {nfr_count}\n\n")
            
            # NFR Subclass distribution
            if nfr_count > 0:
                subclass_counts = {}
                for result in results:
                    if result.nfr_subclass_prediction:
                        subclass = result.nfr_subclass_prediction
                        subclass_counts[subclass] = subclass_counts.get(subclass, 0) + 1
                
                f.write("NFR SUBCLASS DISTRIBUTION:\n")
                for subclass, count in sorted(subclass_counts.items()):
                    percentage = (count / nfr_count) * 100
                    f.write(f"  {subclass}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Performance metrics
            if evaluation_metrics and 'error' not in evaluation_metrics:
                f.write("PERFORMANCE METRICS:\n")
                if 'f1_weighted' in evaluation_metrics:
                    f.write(f"  F1-Score (Weighted): {evaluation_metrics['f1_weighted']:.3f}\n")
                if 'accuracy' in evaluation_metrics:
                    f.write(f"  Accuracy: {evaluation_metrics['accuracy']:.3f}\n")
                if 'hierarchical_accuracy' in evaluation_metrics:
                    f.write(f"  Hierarchical Accuracy: {evaluation_metrics['hierarchical_accuracy']:.3f}\n")
                f.write("\n")
            
            # Processing time statistics
            processing_times = [r.processing_time for r in results]
            f.write("PROCESSING TIME STATISTICS:\n")
            f.write(f"  Mean: {np.mean(processing_times):.2f}s\n")
            f.write(f"  Median: {np.median(processing_times):.2f}s\n")
            f.write(f"  Min: {np.min(processing_times):.2f}s\n")
            f.write(f"  Max: {np.max(processing_times):.2f}s\n")
            f.write(f"  Total: {np.sum(processing_times):.2f}s\n\n")
            
            # Sample classifications
            f.write("SAMPLE CLASSIFICATIONS:\n")
            f.write("-" * 50 + "\n")
            for i, result in enumerate(results[:5]):  # Show first 5
                f.write(f"#{i+1}:\n")
                f.write(f"Text: {result.requirement_text[:80]}...\n")
                f.write(f"Classification: {result.final_classification}\n")
                f.write(f"Confidence: FR/NFR={result.fr_nfr_confidence:.2f}")
                if result.nfr_subclass_confidence:
                    f.write(f", Subclass={result.nfr_subclass_confidence:.2f}")
                f.write("\n")
                f.write("-" * 50 + "\n")
        
        print(f"📄 Summary report saved to: {report_file}")


def load_evaluation_dataset() -> Tuple[List[str], List[str]]:
    """
    Load dataset for evaluation
    
    Returns:
        Tuple of (requirements, labels)
    """
    try:
        # Try to load the subclass dataset
        return load_subclass_dataset()
    except Exception as e:
        print(f"⚠️  Error loading dataset: {e}")
        print("🔄 Using sample data for demonstration")
        return get_sample_nfr_data()


def main():
    """Main function to demonstrate the hierarchical workflow"""
    print("HIERARCHICAL REQUIREMENTS CLASSIFICATION WORKFLOW")
    print("=" * 55)
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not found in environment variables")
        print("Please set your API key: export ANTHROPIC_API_KEY=your_key_here")
        return
    
    # Initialize workflow
    try:
        workflow = HierarchicalRequirementsWorkflow(
            anthropic_api_key=api_key,
            use_few_shot=True,
            temperature=0.0
        )
        print("✅ Hierarchical workflow initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing workflow: {e}")
        return
    
    # Load evaluation data
    print("\n📂 Loading evaluation dataset...")
    requirements, true_labels = load_evaluation_dataset()
    print(f"✅ Loaded {len(requirements)} requirements")
    
    # Run evaluation on a subset for testing
    test_size = min(10, len(requirements))  # Test on first 10 requirements
    test_requirements = requirements[:test_size]
    test_true_labels = true_labels[:test_size]
    
    print(f"\n🧪 Running hierarchical classification on {test_size} requirements...")
    print("This may take a few moments due to API calls...")
    
    # Run the workflow
    results = workflow.classify_batch(
        test_requirements, 
        test_true_labels,
        batch_size=5,
        delay_between_batches=2.0
    )
    
    # Display results
    print("\n📊 HIERARCHICAL CLASSIFICATION RESULTS:")
    print("=" * 55)
    
    for i, result in enumerate(results):
        true_label = test_true_labels[i] if i < len(test_true_labels) else "N/A"
        print(f"\n#{i+1}:")
        print(f"Text: {result.requirement_text[:80]}...")
        print(f"FR/NFR: {result.fr_nfr_prediction} (confidence: {result.fr_nfr_confidence:.2f})")
        
        if result.nfr_subclass_prediction:
            print(f"NFR Subclass: {result.nfr_subclass_prediction} (confidence: {result.nfr_subclass_confidence:.2f})")
        
        print(f"Final Classification: {result.final_classification}")
        print(f"True Label: {true_label}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
        # Show correctness
        if true_label != "N/A":
            if (result.final_classification == true_label or 
                (result.final_classification.endswith(true_label) and true_label in ['O', 'PE', 'SE', 'US'])):
                print("✅ CORRECT")
            else:
                print("❌ INCORRECT")
        
        print("-" * 55)
    
    # Evaluate performance
    if len(test_true_labels) == len(results):
        print("\n📈 EVALUATING PERFORMANCE...")
        
        # Evaluate NFR subclassification
        metrics = workflow.evaluate_hierarchical_performance(
            results, 
            test_true_labels, 
            evaluation_type="subclass_only"
        )
        
        if 'error' not in metrics:
            print("\n🎯 NFR SUBCLASSIFICATION PERFORMANCE:")
            print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"F1-Score (Weighted): {metrics.get('f1_weighted', 0):.3f}")
            print(f"F1-Score (Macro): {metrics.get('f1_macro', 0):.3f}")
            
            print("\n📋 Per-Class Performance:")
            per_class = metrics.get('per_class_metrics', {})
            for class_name, class_metrics in per_class.items():
                print(f"  {class_name}: F1={class_metrics['f1_score']:.3f}, "
                      f"Precision={class_metrics['precision']:.3f}, "
                      f"Recall={class_metrics['recall']:.3f}, "
                      f"Support={class_metrics['support']}")
        else:
            print(f"⚠️  Evaluation limitation: {metrics['error']}")
        
        # Save results
        workflow.save_workflow_results(results, metrics, "demo_test_run")
        print("\n💾 Complete results saved to hierarchical_evaluation_results/ directory")
    
    print("\n✅ HIERARCHICAL WORKFLOW DEMONSTRATION COMPLETED!")
    print("\nNext steps:")
    print("1. Run full evaluation with complete dataset")
    print("2. Compare results with state-of-the-art benchmarks")
    print("3. Integrate with your existing requirements engineering pipeline")


if __name__ == "__main__":
    main()