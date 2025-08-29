#!/usr/bin/env python3
"""
Comprehensive Evaluation of FR/NFR Classification Models

This script evaluates multiple LLM models against the HuggingFace dataset
and compares performance with BERT baselines from Lim (2022).

Sources and Citations:
1. Lim, S. C. (2022). "A Case for Pre-trained Language Models in Systems Engineering". 
   MIT System Design and Management Program. Table 4-10, pp. 69-70.
2. ISO/IEC 25010:2023 - Systems and software Quality Requirements and Evaluation (SQuaRE)
3. IEEE Standards for Requirements Engineering
4. Ferrari, A., et al. (2021). "NLP for Requirements Engineering: Tasks, Techniques, Tools, and Technologies"
5. Zhao, L., et al. (2021). "Natural Language Processing for Requirements Engineering: A Systematic Mapping Study"
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Import our classifier
from langchain_fr_nfr_pipeline import RequirementClassifier, DatasetLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for FR/NFR classification
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # BERT baseline results from Lim (2022), Table 4-10
        self.bert_baselines = {
            "BERT_base": {
                "f1_score": 0.838,
                "precision": 0.838,  # Weighted average from paper
                "recall": 0.838,
                "source": "Lim (2022), Table 4-10"
            },
            "RoBERTa_base": {
                "f1_score": 0.826,
                "precision": 0.826,
                "recall": 0.826,
                "source": "Lim (2022), Table 4-10"
            },
            "SciBERT": {
                "f1_score": 0.850,
                "precision": 0.850,
                "recall": 0.850,
                "source": "Lim (2022), Table 4-10"
            },
            "ReqBERT_TAPT": {
                "f1_score": 0.841,
                "precision": 0.841,
                "recall": 0.841,
                "source": "Lim (2022), Table 4-10 - Task Adaptive Pre-Training"
            },
            "ReqRoBERTa_TAPT": {
                "f1_score": 0.844,
                "precision": 0.844,
                "recall": 0.844,
                "source": "Lim (2022), Table 4-10 - Task Adaptive Pre-Training"
            },
            "ReqSciBERT_TAPT": {
                "f1_score": 0.836,
                "precision": 0.836,
                "recall": 0.836,
                "source": "Lim (2022), Table 4-10 - Task Adaptive Pre-Training"
            }
        }
        
        logger.info(f"Initialized evaluator with output directory: {self.output_dir}")
    
    def load_dataset(self, use_full_dataset: bool = False, sample_size: int = 100) -> Tuple[List[str], List[str]]:
        """
        Load the limsc/fr-nfr-classification dataset
        
        Args:
            use_full_dataset: Whether to use the full dataset (956 requirements)
            sample_size: Number of samples if not using full dataset
            
        Returns:
            Tuple of (requirements, labels)
        """
        try:
            logger.info("Loading limsc/fr-nfr-classification dataset from HuggingFace...")
            dataset = load_dataset("limsc/fr-nfr-classification")
            
            requirements = []
            labels = []
            
            for example in dataset['train']:
                requirements.append(example['reqs'])
                # Convert: is_functional=1 -> "FR", is_functional=0 -> "NFR"
                label = "FR" if example['is_functional'] == 1 else "NFR"
                labels.append(label)
            
            total_size = len(requirements)
            fr_count = labels.count("FR")
            nfr_count = labels.count("NFR")
            
            logger.info(f"Loaded {total_size} requirements")
            logger.info(f"Distribution: FR={fr_count} ({fr_count/total_size:.1%}), NFR={nfr_count} ({nfr_count/total_size:.1%})")
            
            if not use_full_dataset and sample_size < total_size:
                # Stratified sampling to maintain class distribution
                from sklearn.model_selection import train_test_split
                requirements, _, labels, _ = train_test_split(
                    requirements, labels, 
                    train_size=sample_size,
                    stratify=labels,
                    random_state=42
                )
                logger.info(f"Using stratified sample of {len(requirements)} requirements")
            
            return requirements, labels
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            logger.info("Falling back to test samples...")
            return DatasetLoader.create_test_samples()
    
    def evaluate_model_configuration(self, 
                                   model_type: str,
                                   model_name: str,
                                   use_few_shot: bool,
                                   requirements: List[str],
                                   true_labels: List[str],
                                   config_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific model configuration
        
        Args:
            model_type: "claude" or "ollama"
            model_name: Specific model name
            use_few_shot: Whether to use few-shot prompting
            requirements: List of requirements to classify
            true_labels: True labels for evaluation
            config_name: Name for this configuration
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating configuration: {config_name}")
        
        start_time = time.time()
        
        try:
            # Initialize classifier
            classifier = RequirementClassifier(
                model_type=model_type,
                model_name=model_name,
                use_few_shot=use_few_shot,
                temperature=0.1
            )
            
            # Classify requirements
            results = classifier.classify_batch(requirements, true_labels)
            
            # Calculate metrics
            metrics = classifier.evaluate_performance(results)
            
            # Add configuration info
            metrics["configuration"] = {
                "name": config_name,
                "model_type": model_type,
                "model_name": model_name,
                "use_few_shot": use_few_shot,
                "evaluation_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save individual results
            config_dir = self.output_dir / config_name.replace(" ", "_").lower()
            classifier.save_results(results, metrics, str(config_dir))
            
            logger.info(f"Completed {config_name}: F1={metrics.get('f1_score', 0):.3f}, "
                       f"Accuracy={metrics.get('accuracy', 0):.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {config_name}: {str(e)}")
            return {
                "configuration": {"name": config_name, "error": str(e)},
                "f1_score": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
    
    def run_comprehensive_evaluation(self, 
                                   requirements: List[str], 
                                   true_labels: List[str],
                                   model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run evaluation across multiple model configurations
        
        Args:
            requirements: Requirements to classify
            true_labels: True labels
            model_configs: List of model configuration dictionaries
            
        Returns:
            Comprehensive evaluation results
        """
        all_results = {}
        
        logger.info(f"Starting comprehensive evaluation with {len(model_configs)} configurations")
        
        for config in model_configs:
            config_name = config["name"]
            
            try:
                # Skip if API key not available for cloud models
                if config["model_type"] == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
                    logger.warning(f"Skipping {config_name}: ANTHROPIC_API_KEY not set")
                    continue
                
                metrics = self.evaluate_model_configuration(
                    model_type=config["model_type"],
                    model_name=config["model_name"],
                    use_few_shot=config["use_few_shot"],
                    requirements=requirements,
                    true_labels=true_labels,
                    config_name=config_name
                )
                
                all_results[config_name] = metrics
                
            except Exception as e:
                logger.error(f"Failed to evaluate {config_name}: {str(e)}")
                all_results[config_name] = {
                    "configuration": {"name": config_name, "error": str(e)},
                    "f1_score": 0.0
                }
        
        return all_results
    
    def create_comparison_report(self, llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive comparison report with BERT baselines
        
        Args:
            llm_results: Results from LLM evaluations
            
        Returns:
            Comparison report
        """
        logger.info("Creating comparison report...")
        
        # Extract LLM performance data
        llm_performance = {}
        for config_name, results in llm_results.items():
            if "error" not in results.get("configuration", {}):
                llm_performance[config_name] = {
                    "f1_score": results.get("f1_score", 0.0),
                    "precision": results.get("precision", 0.0),
                    "recall": results.get("recall", 0.0),
                    "accuracy": results.get("accuracy", 0.0),
                    "model_type": results.get("configuration", {}).get("model_type", "unknown"),
                    "use_few_shot": results.get("configuration", {}).get("use_few_shot", False)
                }
        
        # Find best performing models
        if llm_performance:
            best_llm = max(llm_performance.items(), key=lambda x: x[1]["f1_score"])
            best_bert = max(self.bert_baselines.items(), key=lambda x: x[1]["f1_score"])
        else:
            best_llm = ("None", {"f1_score": 0.0})
            best_bert = max(self.bert_baselines.items(), key=lambda x: x[1]["f1_score"])
        
        # Create comparison
        comparison_report = {
            "evaluation_metadata": {
                "dataset_source": "limsc/fr-nfr-classification (HuggingFace)",
                "bert_baseline_source": "Lim, S. C. (2022). A Case for Pre-trained Language Models in Systems Engineering. MIT Thesis.",
                "evaluation_date": datetime.now().isoformat(),
                "total_samples": len(llm_results.get(list(llm_results.keys())[0], {}).get("detailed_report", {}).get("macro avg", {})) if llm_results else 0
            },
            
            "bert_baselines": self.bert_baselines,
            
            "llm_results": llm_performance,
            
            "best_performers": {
                "best_llm": {
                    "name": best_llm[0],
                    "metrics": best_llm[1]
                },
                "best_bert": {
                    "name": best_bert[0],
                    "metrics": best_bert[1]
                }
            },
            
            "comparative_analysis": {
                "llm_vs_bert_f1": best_llm[1]["f1_score"] - best_bert[1]["f1_score"],
                "llm_outperforms_bert": best_llm[1]["f1_score"] > best_bert[1]["f1_score"],
                "improvement_percentage": ((best_llm[1]["f1_score"] - best_bert[1]["f1_score"]) / best_bert[1]["f1_score"]) * 100 if best_bert[1]["f1_score"] > 0 else 0
            }
        }
        
        return comparison_report
    
    def create_visualizations(self, comparison_report: Dict[str, Any]):
        """
        Create performance comparison visualizations
        
        Args:
            comparison_report: Comparison report data
        """
        logger.info("Creating visualizations...")
        
        # Prepare data for plotting
        models = []
        f1_scores = []
        model_types = []
        
        # Add BERT baselines
        for name, metrics in comparison_report["bert_baselines"].items():
            models.append(name)
            f1_scores.append(metrics["f1_score"])
            model_types.append("BERT")
        
        # Add LLM results
        for name, metrics in comparison_report["llm_results"].items():
            models.append(name)
            f1_scores.append(metrics["f1_score"])
            model_types.append("LLM")
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        colors = ['#1f77b4' if mt == 'BERT' else '#ff7f0e' for mt in model_types]
        
        bars = plt.barh(models, f1_scores, color=colors, alpha=0.7)
        plt.xlabel('F1-Score')
        plt.title('FR/NFR Classification Performance: LLM vs BERT Models\n' + 
                 'Based on Lim (2022) BERT baselines and current LLM evaluation')
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=9)
        
        # Add legend
        bert_patch = plt.Rectangle((0, 0), 1, 1, fc='#1f77b4', alpha=0.7)
        llm_patch = plt.Rectangle((0, 0), 1, 1, fc='#ff7f0e', alpha=0.7)
        plt.legend([bert_patch, llm_patch], ['BERT Models', 'LLM Models'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create performance metrics heatmap for LLMs
        if comparison_report["llm_results"]:
            llm_data = []
            llm_names = []
            
            for name, metrics in comparison_report["llm_results"].items():
                llm_names.append(name.replace(" ", "\n"))
                llm_data.append([
                    metrics["accuracy"],
                    metrics["precision"], 
                    metrics["recall"],
                    metrics["f1_score"]
                ])
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                np.array(llm_data).T,
                annot=True,
                fmt='.3f',
                xticklabels=llm_names,
                yticklabels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                cmap='RdYlGn',
                cbar_kws={'label': 'Score'}
            )
            plt.title('LLM Performance Metrics Heatmap')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'llm_metrics_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def generate_final_report(self, comparison_report: Dict[str, Any]):
        """
        Generate final evaluation report
        """
        logger.info("Generating final report...")
        
        # Save detailed JSON report
        report_file = self.output_dir / 'comprehensive_evaluation_report.json'
        with open(report_file, 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        # Generate markdown summary
        md_report = self._generate_markdown_report(comparison_report)
        md_file = self.output_dir / 'evaluation_summary.md'
        with open(md_file, 'w') as f:
            f.write(md_report)
        
        logger.info(f"Final report saved to {report_file}")
        logger.info(f"Summary report saved to {md_file}")
    
    def _generate_markdown_report(self, comparison_report: Dict[str, Any]) -> str:
        """
        Generate markdown summary report
        """
        best_llm = comparison_report["best_performers"]["best_llm"]
        best_bert = comparison_report["best_performers"]["best_bert"]
        analysis = comparison_report["comparative_analysis"]
        
        md_content = f"""# FR/NFR Classification Evaluation Report

## Executive Summary

This report presents a comprehensive evaluation of Large Language Models (LLMs) for classifying software requirements into Functional Requirements (FR) and Non-Functional Requirements (NFR), compared against BERT-based baselines.

**Key Findings:**
- Best LLM: {best_llm['name']} (F1-Score: {best_llm['metrics']['f1_score']:.3f})
- Best BERT: {best_bert['name']} (F1-Score: {best_bert['metrics']['f1_score']:.3f})
- Performance Difference: {analysis['llm_vs_bert_f1']:+.3f} ({analysis['improvement_percentage']:+.1f}%)
- LLM Outperforms BERT: {'Yes' if analysis['llm_outperforms_bert'] else 'No'}

## Data Sources and Citations

### Primary Sources:
1. **Lim, S. C. (2022).** "A Case for Pre-trained Language Models in Systems Engineering." 
   *MIT System Design and Management Program.* 
   - BERT baseline results from Table 4-10 (pp. 69-70)

2. **Dataset:** `limsc/fr-nfr-classification` (HuggingFace)
   - {comparison_report['evaluation_metadata'].get('total_samples', 'N/A')} samples evaluated

3. **Standards:**
   - ISO/IEC 25010:2023 - Systems and software Quality Requirements and Evaluation (SQuaRE)
   - IEEE Standards for Requirements Engineering

### Supporting Literature:
- Ferrari, A., et al. (2021). "NLP for Requirements Engineering: Tasks, Techniques, Tools, and Technologies"
- Zhao, L., et al. (2021). "Natural Language Processing for Requirements Engineering: A Systematic Mapping Study"
- Chen, L., Babar, M. A., & Nuseibeh, B. (2013). "Characterizing Architecturally Significant Requirements". *IEEE Software.*
- Glinz, M. (2008). "A Risk-Based, Value-Oriented Approach to Quality Requirements". *IEEE Software.*

## Performance Results

### BERT Baselines (from Lim, 2022)
"""

        # Add BERT results table
        md_content += "\n| Model | F1-Score | Source |\n|-------|----------|--------|\n"
        for name, metrics in comparison_report["bert_baselines"].items():
            md_content += f"| {name} | {metrics['f1_score']:.3f} | {metrics['source']} |\n"

        # Add LLM results if available
        if comparison_report["llm_results"]:
            md_content += "\n### LLM Results (Current Evaluation)\n\n"
            md_content += "| Model Configuration | F1-Score | Precision | Recall | Accuracy | Few-Shot |\n"
            md_content += "|-------------------|----------|-----------|---------|----------|----------|\n"
            
            for name, metrics in comparison_report["llm_results"].items():
                few_shot = "Yes" if metrics.get("use_few_shot", False) else "No"
                md_content += f"| {name} | {metrics['f1_score']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['accuracy']:.3f} | {few_shot} |\n"

        md_content += f"""

## Analysis

### Performance Comparison
- **Best LLM Performance:** {best_llm['metrics']['f1_score']:.3f} F1-Score ({best_llm['name']})
- **Best BERT Performance:** {best_bert['metrics']['f1_score']:.3f} F1-Score ({best_bert['name']})
- **Difference:** {analysis['llm_vs_bert_f1']:+.3f} ({analysis['improvement_percentage']:+.1f}% change)

### Key Observations

1. **Classification Approach:** LLMs use natural language understanding and few-shot learning, while BERT models require fine-tuning on domain-specific data.

2. **Interpretability:** LLM classifications include reasoning and key indicators, providing explainable results compared to BERT's black-box approach.

3. **Flexibility:** LLMs can adapt to new domains without retraining, while BERT models require Task-Adaptive Pre-Training (TAPT) for optimal performance.

4. **Resource Requirements:** LLMs require API access or local deployment, while BERT models need training infrastructure and domain-specific datasets.

## Methodology

### Classification Rules
The evaluation used comprehensive classification rules derived from:
- ISO/IEC 25010:2023 quality characteristics
- IEEE Requirements Engineering standards
- Linguistic patterns identified in academic literature

### Functional Requirements (FR) Indicators:
- Action verbs: perform, execute, calculate, process, display, store
- Behavioral descriptions: "system shall", "must execute"
- Data manipulation operations

### Non-Functional Requirements (NFR) Indicators:
- Quality attributes: performance, security, usability, reliability
- Constraints: time limits, resource constraints, compliance requirements
- ISO/IEC 25010 characteristics: efficiency, maintainability, portability

## Implications for Practice

### When to Use LLMs:
- **Exploratory analysis** of new requirement documents
- **Rapid classification** without training data
- **Explainable results** needed for stakeholder communication
- **Cross-domain** requirements analysis

### When to Use BERT:
- **High-volume production** classification
- **Domain-specific optimization** with available training data
- **Consistent performance** requirements
- **Resource-constrained** environments

## Limitations

1. **Sample Size:** Evaluation based on limited dataset samples
2. **Domain Specificity:** Results may vary for different engineering domains
3. **Cost Considerations:** LLM API costs vs. BERT training costs not evaluated
4. **Temporal Stability:** LLM performance may vary with model updates

## Recommendations

1. **Hybrid Approach:** Use LLMs for initial classification and explanation, BERT for high-volume production
2. **Domain Adaptation:** Develop domain-specific prompts for specialized engineering fields
3. **Continuous Evaluation:** Regular assessment as both LLM and BERT models evolve
4. **Tool Integration:** Incorporate both approaches in requirements management tools

---

*Report generated on: {comparison_report['evaluation_metadata']['evaluation_date']}*

*For detailed results and raw data, see the accompanying JSON report and CSV files.*
"""
        
        return md_content


def main():
    """
    Main evaluation function
    """
    parser = argparse.ArgumentParser(description='Comprehensive FR/NFR Classification Evaluation')
    parser.add_argument('--full-dataset', action='store_true', 
                       help='Use full dataset (956 samples) instead of subset')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Sample size if not using full dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(output_dir=args.output_dir)
    
    # Load dataset
    try:
        requirements, labels = evaluator.load_dataset(
            use_full_dataset=args.full_dataset,
            sample_size=args.sample_size
        )
        logger.info(f"Loaded {len(requirements)} requirements for evaluation")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Using test samples instead")
        requirements, labels = DatasetLoader.create_test_samples()
    
    # Define model configurations to test
    model_configs = [
        # {
        #     "name": "Claude-4-Sonnet (Few-Shot)",
        #     "model_type": "claude",
        #     "model_name": "claude-sonnet-4-20250514",
        #     "use_few_shot": True
        # },
        # {
        #     "name": "Claude-4-Sonnet (Zero-Shot)",
        #     "model_type": "claude", 
        #     "model_name": "claude-sonnet-4-20250514",
        #     "use_few_shot": False
        # },
        # {
        #     "name": "Claude-3.5-Haiku (Few-Shot)",
        #     "model_type": "claude",
        #     "model_name": "claude-3-5-haiku-20241022",
        #     "use_few_shot": True
        # },
        # {
        #     "name": "Claude-3.5-Haiku (Zero-Shot)",
        #     "model_type": "claude",
        #     "model_name": "claude-3-5-haiku-20241022",
        #     "use_few_shot": False
        # },
        {
            "name": "DeepSeek R1 (Few-Shot)",
            "model_type": "openai",
            "model_name": "deepseek-chat",
            "use_few_shot": True
        },
    ]
    
    # Add Ollama models if available
    ollama_models = [
        # {
        #     "name": "Llama-3.2-3B (Few-Shot)",
        #     "model_type": "ollama",
        #     "model_name": "llama3.2:latest",
        #     "use_few_shot": True
        # },
        # {
        #     "name": "Mistral-7B (Few-Shot)", 
        #     "model_type": "ollama",
        #     "model_name": "mistral:7b",
        #     "use_few_shot": True
        # },
        # {
        #     "name": "Qwen3-4B (Few-Shot)",
        #     "model_type": "ollama", 
        #     "model_name": "qwen3:4b",
        #     "use_few_shot": True
        # },
        # {
        #     "name": "DeepSeek-R1-8B (Few-Shot)",
        #     "model_type": "ollama", 
        #     "model_name": "deepseek-r1:8b",
        #     "use_few_shot": True
        # },
        # {
        #     "name": "DeepSeek-R1-8B (Few-Shot)",
        #     "model_type": "ollama", 
        #     "model_name": "deepseek-r1:8b",
        #     "use_few_shot": False
        # },
    ]
    
    # Check if Ollama is available
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama server detected, adding local models")
            model_configs.extend(ollama_models)
        else:
            logger.info("Ollama server not available, skipping local models")
    except:
        logger.info("Ollama server not available, skipping local models")
    
    logger.info(f"Testing {len(model_configs)} model configurations")
    
    # Run comprehensive evaluation
    try:
        llm_results = evaluator.run_comprehensive_evaluation(
            requirements=requirements,
            true_labels=labels,
            model_configs=model_configs
        )
        
        # Create comparison report
        comparison_report = evaluator.create_comparison_report(llm_results)
        
        # Generate visualizations
        evaluator.create_visualizations(comparison_report)
        
        # Generate final report
        evaluator.generate_final_report(comparison_report)
        
        # Print summary
        best_llm = comparison_report["best_performers"]["best_llm"]
        best_bert = comparison_report["best_performers"]["best_bert"]
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Best LLM:  {best_llm['name']} (F1: {best_llm['metrics']['f1_score']:.3f})")
        print(f"Best BERT: {best_bert['name']} (F1: {best_bert['metrics']['f1_score']:.3f})")
        print(f"Improvement: {comparison_report['comparative_analysis']['llm_vs_bert_f1']:+.3f}")
        print(f"LLM Outperforms BERT: {comparison_report['comparative_analysis']['llm_outperforms_bert']}")
        print(f"\nDetailed results saved to: {evaluator.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example environment setup (uncomment and set your API keys)
    # os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key-here"
    os.getenv("ANTHROPIC_API_KEY")
    
    main()