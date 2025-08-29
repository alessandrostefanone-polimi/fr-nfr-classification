"""
Fixed Benchmark Evaluation Module for Complete Pipeline Testing
==============================================================

This module provides comprehensive evaluation of the complete hierarchical pipeline:
1. FR/NFR Classification (using your existing langchain_fr_nfr_pipeline)
2. NFR Subclassification (using LLM-based approach)

Integrates with your existing repository structure and evaluates against
state-of-the-art results from academic literature.

Author: Alessandro Stefanone
Affiliation: Politecnico di Milano - PhD in Mechanical Engineering
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import your existing FR/NFR classifier
try:
    from langchain_fr_nfr_pipeline import RequirementClassifier
    print("✅ Successfully imported existing FR/NFR classifier")
except ImportError as e:
    print(f"❌ Could not import existing FR/NFR classifier: {e}")
    print("Please ensure langchain_fr_nfr_pipeline.py is in the current directory")
    sys.exit(1)

# Import NFR subclassification components
try:
    from nfr_subclass_classifier import (
        LLMNFRSubclassifier, 
        NFRSubclassResult,
        get_sample_nfr_data
    )
    print("✅ Successfully imported NFR subclassification components")
except ImportError as e:
    print(f"❌ Could not import NFR subclassification components: {e}")
    print("Please ensure nfr_subclass_classifier.py is in the current directory")
    sys.exit(1)

# Dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️  Datasets library not available. Using sample data.")
    DATASETS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StateOfArtBenchmarks:
    """State-of-the-art benchmark results from literature"""
    
    # Baseline results from Lim, S.C. (2022) MIT Thesis
    # ReqBERT (Task-Adaptive BERT) results for NFR subclassification
    REQBERT_RESULTS = {
        'O': {   # Operational
            'precision': 0.71,
            'recall': 0.75,
            'f1_score': 0.73,
            'support': 45
        },
        'PE': {  # Performance
            'precision': 0.96,
            'recall': 0.94,
            'f1_score': 0.95,
            'support': 85
        },
        'SE': {  # Security
            'precision': 0.83,
            'recall': 0.79,
            'f1_score': 0.81,
            'support': 65
        },
        'US': {  # Usability
            'precision': 0.82,
            'recall': 0.80,
            'f1_score': 0.81,
            'support': 55
        }
    }
    
    # FR/NFR classification benchmarks
    FR_NFR_RESULTS = {
        'FR': {
            'precision': 0.92,
            'recall': 0.94,
            'f1_score': 0.93,
            'support': 400
        },
        'NFR': {
            'precision': 0.89,
            'recall': 0.87,
            'f1_score': 0.88,
            'support': 300
        }
    }
    
    @classmethod
    def get_nfr_benchmarks(cls) -> Dict:
        """Get benchmarks for NFR subclassification"""
        nfr_classes = ['O', 'PE', 'SE', 'US']
        nfr_results = {k: v for k, v in cls.REQBERT_RESULTS.items() if k in nfr_classes}
        
        # Calculate overall metrics
        total_support = sum(metrics['support'] for metrics in nfr_results.values())
        weighted_f1 = sum(
            metrics['f1_score'] * metrics['support'] 
            for metrics in nfr_results.values()
        ) / total_support
        
        macro_f1 = np.mean([metrics['f1_score'] for metrics in nfr_results.values()])
        
        return {
            'per_class': nfr_results,
            'f1_weighted': weighted_f1,
            'f1_macro': macro_f1,
            'total_samples': total_support
        }


class CompleteHierarchicalPipeline:
    """
    Complete pipeline integrating FR/NFR classification with NFR subclassification
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """
        Initialize the complete hierarchical pipeline
        
        Args:
            anthropic_api_key: API key for Anthropic Claude
        """
        
        # Initialize FR/NFR classifier (your existing implementation)
        self.fr_nfr_classifier = RequirementClassifier(
            model_type="claude",
            model_name="claude-3-sonnet-20240229",
            use_few_shot=True
        )
        
        # Initialize NFR subclassifier
        self.nfr_subclassifier = LLMNFRSubclassifier(
            api_key=anthropic_api_key,
            use_few_shot=True,
            temperature=0.0
        )
        
        logger.info("Complete hierarchical pipeline initialized")
    
    def classify_requirement(self, requirement_text: str) -> Dict:
        """
        Classify a requirement through the complete hierarchical pipeline
        
        Args:
            requirement_text: The requirement text to classify
            
        Returns:
            Dictionary with complete classification results
        """
        start_time = datetime.now()
        
        # Step 1: FR/NFR Classification
        fr_nfr_result = self.fr_nfr_classifier.classify_requirement(requirement_text)
        
        result = {
            'requirement_text': requirement_text,
            'fr_nfr_prediction': fr_nfr_result.predicted_label,
            'fr_nfr_confidence': fr_nfr_result.confidence,
            'fr_nfr_reasoning': fr_nfr_result.reasoning,
            'nfr_subclass_prediction': None,
            'nfr_subclass_confidence': None,
            'nfr_subclass_reasoning': None,
            'final_classification': fr_nfr_result.predicted_label,
            'processing_time': 0.0
        }
        
        # Step 2: NFR Subclassification (if NFR)
        if fr_nfr_result.predicted_label == "NFR":
            nfr_result = self.nfr_subclassifier.classify_requirement(requirement_text)
            
            result.update({
                'nfr_subclass_prediction': nfr_result.predicted_label,
                'nfr_subclass_confidence': nfr_result.confidence,
                'nfr_subclass_reasoning': nfr_result.reasoning,
                'final_classification': f"NFR_{nfr_result.predicted_label}"
            })
        
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def classify_batch(self, requirements: List[str], batch_size: int = 5) -> List[Dict]:
        """
        Classify a batch of requirements
        
        Args:
            requirements: List of requirement texts
            batch_size: Batch size for rate limiting
            
        Returns:
            List of classification results
        """
        results = []
        total = len(requirements)
        
        logger.info(f"Starting batch classification of {total} requirements")
        
        for i, requirement in enumerate(requirements):
            result = self.classify_requirement(requirement)
            results.append(result)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{total} requirements")
            
            # Rate limiting
            if (i + 1) % batch_size == 0 and i + 1 < total:
                import time
                time.sleep(1.5)  # Respectful API usage
        
        logger.info(f"Batch classification completed: {len(results)} results")
        return results


class PipelineBenchmarkEvaluator:
    """Comprehensive benchmark evaluator for the complete pipeline"""
    
    def __init__(self, output_dir: str = "pipeline_benchmark_results"):
        """Initialize evaluator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmarks = StateOfArtBenchmarks()
        
        # Class mappings
        self.nfr_classes = ['O', 'PE', 'SE', 'US']
        self.class_full_names = {
            'O': 'Operational',
            'PE': 'Performance',
            'SE': 'Security',
            'US': 'Usability'
        }
    
    def run_complete_evaluation(
        self,
        pipeline: CompleteHierarchicalPipeline,
        sample_size: Optional[int] = None
    ) -> Dict:
        """
        Run complete pipeline evaluation
        
        Args:
            pipeline: The complete hierarchical pipeline
            sample_size: Number of samples to evaluate (None for all available)
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting complete pipeline evaluation")
        
        # Load evaluation data
        requirements, true_labels = self._load_evaluation_data()
        
        # Limit sample size if specified
        if sample_size and sample_size < len(requirements):
            requirements = requirements[:sample_size]
            true_labels = true_labels[:sample_size]
        
        logger.info(f"Evaluating on {len(requirements)} requirements")
        
        # Run classification
        results = pipeline.classify_batch(requirements)
        
        # Add true labels to results
        for i, result in enumerate(results):
            if i < len(true_labels):
                result['true_label'] = true_labels[i]
        
        # Evaluate performance
        evaluation = self._evaluate_results(results)
        
        # Save results
        self._save_evaluation_results(results, evaluation)
        
        return evaluation
    
    def _load_evaluation_data(self) -> Tuple[List[str], List[str]]:
        """Load evaluation dataset"""
        try:
            # Try to load from HuggingFace dataset
            if DATASETS_AVAILABLE:
                dataset = load_dataset("limsc/subclass-classification")
                if 'train' in dataset:
                    data = dataset['train']
                else:
                    data = dataset[list(dataset.keys())[0]]
                
                df = data.to_pandas()
                requirements = df.iloc[:, 0].tolist()  # First column assumed to be text
                labels = df.iloc[:, -1].tolist()      # Last column assumed to be labels
                
                logger.info(f"Loaded {len(requirements)} requirements from HuggingFace dataset")
                return requirements, labels
        except Exception as e:
            logger.warning(f"Could not load HuggingFace dataset: {e}")
        
        # Fallback to sample data
        logger.info("Using sample data for evaluation")
        return get_sample_nfr_data()
    
    def _evaluate_results(self, results: List[Dict]) -> Dict:
        """Evaluate pipeline results"""
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(results),
            'fr_nfr_evaluation': self._evaluate_fr_nfr(results),
            'nfr_subclass_evaluation': self._evaluate_nfr_subclass(results),
            'pipeline_stats': self._calculate_pipeline_stats(results)
        }
        
        return evaluation
    
    def _evaluate_fr_nfr(self, results: List[Dict]) -> Dict:
        """Evaluate FR/NFR classification performance"""
        # Extract FR/NFR predictions and true labels
        predictions = []
        true_labels = []
        
        for result in results:
            if 'true_label' in result:
                # Convert NFR subclass labels to just NFR
                true_label = result['true_label']
                if true_label in ['O', 'PE', 'SE', 'US']:
                    true_label = 'NFR'
                
                predictions.append(result['fr_nfr_prediction'])
                true_labels.append(true_label)
        
        if not predictions:
            return {'error': 'No valid FR/NFR predictions found'}
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=['FR', 'NFR']
        )
        
        return {
            'accuracy': accuracy,
            'fr_metrics': {
                'precision': precision[0] if len(precision) > 0 else 0,
                'recall': recall[0] if len(recall) > 0 else 0,
                'f1_score': f1[0] if len(f1) > 0 else 0,
                'support': int(support[0]) if len(support) > 0 else 0
            },
            'nfr_metrics': {
                'precision': precision[1] if len(precision) > 1 else 0,
                'recall': recall[1] if len(recall) > 1 else 0,
                'f1_score': f1[1] if len(f1) > 1 else 0,
                'support': int(support[1]) if len(support) > 1 else 0
            }
        }
    
    def _evaluate_nfr_subclass(self, results: List[Dict]) -> Dict:
        """Evaluate NFR subclassification performance"""
        # Extract NFR subclass predictions
        predictions = []
        true_labels = []
        
        for result in results:
            if (result.get('fr_nfr_prediction') == 'NFR' and 
                result.get('nfr_subclass_prediction') and 
                'true_label' in result):
                
                true_label = result['true_label']
                if true_label in self.nfr_classes:
                    predictions.append(result['nfr_subclass_prediction'])
                    true_labels.append(true_label)
        
        if not predictions:
            return {'error': 'No valid NFR subclass predictions found'}
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=self.nfr_classes, zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.nfr_classes):
            per_class_metrics[class_name] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1_score': f1[i] if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        # Overall metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Compare with benchmarks
        benchmarks = self.benchmarks.get_nfr_benchmarks()
        comparison = self._compare_with_benchmarks(per_class_metrics, benchmarks)
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_metrics': per_class_metrics,
            'benchmark_comparison': comparison,
            'classification_report': classification_report(
                true_labels, predictions, labels=self.nfr_classes, zero_division=0
            )
        }
    
    def _compare_with_benchmarks(self, llm_metrics: Dict, benchmarks: Dict) -> Dict:
        """Compare LLM performance with benchmarks"""
        comparison = {
            'overall_comparison': {},
            'per_class_comparison': {},
            'performance_summary': {}
        }
        
        # Overall comparison
        llm_f1_weighted = sum(
            llm_metrics[cls]['f1_score'] * llm_metrics[cls]['support']
            for cls in self.nfr_classes if cls in llm_metrics and llm_metrics[cls]['support'] > 0
        ) / max(1, sum(llm_metrics[cls]['support'] for cls in self.nfr_classes if cls in llm_metrics))
        
        benchmark_f1_weighted = benchmarks['f1_weighted']
        
        comparison['overall_comparison'] = {
            'llm_f1_weighted': llm_f1_weighted,
            'benchmark_f1_weighted': benchmark_f1_weighted,
            'improvement': ((llm_f1_weighted - benchmark_f1_weighted) / benchmark_f1_weighted) * 100
        }
        
        # Per-class comparison
        improvements = []
        for class_name in self.nfr_classes:
            if class_name in llm_metrics and class_name in benchmarks['per_class']:
                llm_f1 = llm_metrics[class_name]['f1_score']
                benchmark_f1 = benchmarks['per_class'][class_name]['f1_score']
                
                improvement = ((llm_f1 - benchmark_f1) / benchmark_f1) * 100 if benchmark_f1 > 0 else 0
                improvements.append(improvement)
                
                comparison['per_class_comparison'][class_name] = {
                    'class_full_name': self.class_full_names[class_name],
                    'llm_f1': llm_f1,
                    'benchmark_f1': benchmark_f1,
                    'improvement': improvement
                }
        
        # Performance summary
        comparison['performance_summary'] = {
            'classes_improved': len([imp for imp in improvements if imp > 0]),
            'classes_degraded': len([imp for imp in improvements if imp < 0]),
            'average_improvement': np.mean(improvements) if improvements else 0,
            'overall_competitive': llm_f1_weighted >= (benchmark_f1_weighted * 0.95)
        }
        
        return comparison
    
    def _calculate_pipeline_stats(self, results: List[Dict]) -> Dict:
        """Calculate pipeline statistics"""
        processing_times = [r['processing_time'] for r in results]
        fr_count = len([r for r in results if r['fr_nfr_prediction'] == 'FR'])
        nfr_count = len([r for r in results if r['fr_nfr_prediction'] == 'NFR'])
        
        # NFR subclass distribution
        nfr_subclass_counts = {}
        for result in results:
            if result.get('nfr_subclass_prediction'):
                subclass = result['nfr_subclass_prediction']
                nfr_subclass_counts[subclass] = nfr_subclass_counts.get(subclass, 0) + 1
        
        return {
            'fr_count': fr_count,
            'nfr_count': nfr_count,
            'nfr_subclass_distribution': nfr_subclass_counts,
            'processing_time_stats': {
                'mean': np.mean(processing_times),
                'median': np.median(processing_times),
                'min': np.min(processing_times),
                'max': np.max(processing_times),
                'total': np.sum(processing_times)
            }
        }
    
    def _save_evaluation_results(self, results: List[Dict], evaluation: Dict):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_file = self.output_dir / f"pipeline_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        
        # Save evaluation metrics
        metrics_file = self.output_dir / f"evaluation_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate summary report
        self._generate_summary_report(evaluation, timestamp)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_summary_report(self, evaluation: Dict, timestamp: str):
        """Generate summary report"""
        report_file = self.output_dir / f"benchmark_summary_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Complete Pipeline Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Requirements Processed:** {evaluation['total_samples']}\n")
            
            # FR/NFR Performance
            if 'error' not in evaluation['fr_nfr_evaluation']:
                fr_nfr = evaluation['fr_nfr_evaluation']
                f.write(f"- **FR/NFR Classification Accuracy:** {fr_nfr['accuracy']:.3f}\n")
            
            # NFR Subclassification Performance
            if 'error' not in evaluation['nfr_subclass_evaluation']:
                nfr_eval = evaluation['nfr_subclass_evaluation']
                f.write(f"- **NFR Subclassification F1 (Weighted):** {nfr_eval['f1_weighted']:.3f}\n")
                
                if 'benchmark_comparison' in nfr_eval:
                    comp = nfr_eval['benchmark_comparison']['overall_comparison']
                    f.write(f"- **Improvement vs State-of-Art:** {comp['improvement']:+.1f}%\n")
            
            # Pipeline Statistics
            stats = evaluation['pipeline_stats']
            f.write(f"- **FR Requirements:** {stats['fr_count']}\n")
            f.write(f"- **NFR Requirements:** {stats['nfr_count']}\n")
            f.write(f"- **Average Processing Time:** {stats['processing_time_stats']['mean']:.2f}s\n")
            
            # Detailed Performance
            if 'error' not in evaluation['nfr_subclass_evaluation'] and 'benchmark_comparison' in evaluation['nfr_subclass_evaluation']:
                f.write("\n## NFR Subclassification Performance\n\n")
                f.write("| Class | LLM F1 | Benchmark F1 | Improvement |\n")
                f.write("|-------|--------|--------------|-------------|\n")
                
                per_class_comp = evaluation['nfr_subclass_evaluation']['benchmark_comparison']['per_class_comparison']
                for class_name, comp_data in per_class_comp.items():
                    f.write(f"| {class_name} ({comp_data['class_full_name']}) | "
                           f"{comp_data['llm_f1']:.3f} | {comp_data['benchmark_f1']:.3f} | "
                           f"{comp_data['improvement']:+.1f}% |\n")
        
        logger.info(f"Summary report generated: {report_file}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Complete Pipeline Benchmark Evaluation")
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--api-key', type=str,
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get API key
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        print("Please set the environment variable or use --api-key argument")
        return
    
    print("COMPLETE PIPELINE BENCHMARK EVALUATION")
    print("=" * 50)
    print(f"Sample Size: {args.samples}")
    print(f"API Key: {'***' + api_key[-4:] if api_key else 'Not provided'}")
    
    try:
        # Initialize pipeline
        print("\n🔧 Initializing complete hierarchical pipeline...")
        pipeline = CompleteHierarchicalPipeline(anthropic_api_key=api_key)
        
        # Initialize evaluator
        evaluator = PipelineBenchmarkEvaluator()
        
        print("✅ Pipeline initialized successfully")
        
        # Run evaluation
        print(f"\n🚀 Running evaluation on {args.samples} samples...")
        print("⏱️  This may take several minutes...")
        
        results = evaluator.run_complete_evaluation(pipeline, sample_size=args.samples)
        
        # Display summary
        print("\n📊 EVALUATION COMPLETED!")
        print("=" * 30)
        
        if 'error' not in results['nfr_subclass_evaluation']:
            nfr_eval = results['nfr_subclass_evaluation']
            print(f"NFR Subclass F1 (Weighted): {nfr_eval['f1_weighted']:.3f}")
            
            if 'benchmark_comparison' in nfr_eval:
                comp = nfr_eval['benchmark_comparison']['overall_comparison']
                print(f"Improvement vs ReqBERT: {comp['improvement']:+.1f}%")
        
        stats = results['pipeline_stats']
        print(f"Processing Time (avg): {stats['processing_time_stats']['mean']:.2f}s")
        print(f"Total Requirements: {results['total_samples']}")
        
        print(f"\n💾 Detailed results saved to: pipeline_benchmark_results/")
        print("🎉 Complete pipeline evaluation finished!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()