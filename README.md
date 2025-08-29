# Hierarchical Requirements Classification AI Workflow

An advanced AI workflow for classifying software/system requirements using a hierarchical approach that combines your existing FR/NFR classifier with LLM-based NFR subclassification.

## 🎯 Objective

This implementation supports the research objective of **developing an end-to-end semantic features extraction from engineering requirements** contained in technical documents, enabling real-time monitoring and predictive maintenance through Digital Twin technologies.

## 🏗️ Architecture

```
Input Requirement Text
        ↓
FR/NFR Classification (Existing Classifier)
        ↓
    [if NFR]
        ↓
LLM-based NFR Subclassification
  ├── Operational (O)
  ├── Performance (PE) 
  ├── Security (SE)
  └── Usability (US)
        ↓
Final Hierarchical Label
```

## 📚 Academic Foundation

### Primary Sources:
- **Lim, S.C. (2022)** - "A Case for Pre-trained Language Models in Systems Engineering" (MIT Thesis)
- **ISO/IEC 25010:2023** - Systems and software Quality Requirements and Evaluation (SQuaRE)
- **IEEE Standards** for Requirements Engineering

### Dataset:
- **HuggingFace:** `limsc/subclass-classification` for NFR subclassification
- **Integration** with existing `alessandrostefanone-polimi/fr-nfr-classification`

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone or download the project files
git clone <your-repo-url>
cd hierarchical-requirements-classification

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.template .env
# Edit .env with your Anthropic API key
```

### 2. Run Demonstration
```bash
# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# Run demo
python demo.py
```

### 3. Run Full Benchmark Evaluation
```bash
# Compare against state-of-the-art (ReqBERT)
python benchmark_evaluation.py
```

## 📁 Project Structure

```
├── nfr_subclass_classifier.py      # Core LLM-based NFR subclassification
├── hierarchical_workflow.py        # Complete workflow integration  
├── benchmark_evaluation.py         # State-of-the-art comparison
├── demo.py                         # Demonstration script
├── requirements.txt                # Python dependencies
├── .env.template                   # Environment variables template
├── README.md                       # This file
├── evaluation_results/             # Classification results
├── benchmark_results/              # Benchmark comparisons
└── logs/                          # Application logs
```

## 🔧 Core Components

### 1. NFR Subclassification Module (`nfr_subclass_classifier.py`)
- **LLMNFRSubclassifier**: Main classification engine using Anthropic Claude
- **Prompt Engineering**: Literature-based classification rules
- **Few-shot Learning**: Examples derived from academic standards
- **Structured Output**: Pydantic models for consistent results

### 2. Hierarchical Workflow (`hierarchical_workflow.py`)  
- **Integration**: Combines FR/NFR + NFR subclassification
- **Batch Processing**: Efficient handling of multiple requirements
- **Rate Limiting**: Respectful API usage with delays
- **Result Management**: Comprehensive result storage and analysis

### 3. Benchmark Evaluation (`benchmark_evaluation.py`)
- **State-of-Art Comparison**: Against ReqBERT (MIT thesis baselines)
- **Performance Metrics**: Precision, Recall, F1-scores
- **Visualizations**: Charts and confusion matrices
- **Detailed Reports**: Markdown reports with insights

## 📊 Expected Performance

Based on ReqBERT baselines from Lim, S.C. (2022):

| NFR Subclass | ReqBERT F1 | Target F1 | Description |
|-------------|------------|-----------|-------------|
| Operational (O) | 0.73 | ≥0.73 | System operation, monitoring |
| Performance (PE) | 0.95 | ≥0.95 | Speed, throughput, scalability |
| Security (SE) | 0.81 | ≥0.81 | Authentication, encryption |  
| Usability (US) | 0.81 | ≥0.81 | User interface, accessibility |
| **Overall** | **0.866** | **≥0.85** | **Weighted F1-Score** |

## 💡 Key Features

### ✅ Advantages of LLM Approach
- **Zero-shot Capability**: No training data required
- **Interpretable Results**: Reasoning and key indicators provided
- **Rapid Deployment**: API-based, no model training infrastructure
- **Easy Updates**: Prompt engineering for improvements
- **Multilingual Support**: Inherent in large language models

### ⚠️ Considerations
- **API Dependency**: Requires Anthropic API access
- **Cost Structure**: Per-request pricing model
- **Network Latency**: API calls vs local inference
- **Rate Limits**: Requires batch processing with delays

## 🧪 Usage Examples

### Single Requirement Classification
```python
from hierarchical_workflow import HierarchicalRequirementsWorkflow

workflow = HierarchicalRequirementsWorkflow(
    anthropic_api_key="your-key-here",
    use_few_shot=True
)

result = workflow.classify_single_requirement(
    "The system shall respond within 2 seconds under normal load"
)

print(f"Classification: {result.final_classification}")
print(f"Reasoning: {result.reasoning}")
```

### Batch Processing
```python
requirements = ["req1", "req2", "req3"]
results = workflow.classify_batch(
    requirements,
    batch_size=5,
    delay_between_batches=1.0
)

# Evaluate performance
metrics = workflow.evaluate_hierarchical_performance(
    results, true_labels, evaluation_type="subclass_only"
)
```

### Benchmark Against State-of-Art
```python
from benchmark_evaluation import BenchmarkEvaluator

evaluator = BenchmarkEvaluator()
benchmark_results = evaluator.run_comprehensive_benchmark(
    workflow, dataset_size=100
)
```

## 📈 Integration with Digital Twins

This workflow supports your broader PhD research objectives:

1. **Semantic Feature Extraction**: Hierarchical classification provides structured requirement features
2. **Real-time Monitoring**: API-based classification enables live requirement analysis  
3. **Product Lifecycle Management**: Integration with PLM systems through structured outputs
4. **Predictive Maintenance**: NFR subclassification supports system health monitoring

## 🔬 Research Applications

### For Academic Publication (Q1 Journal Ready):
- **Novel Methodology**: LLM-based hierarchical classification for requirements engineering
- **Comprehensive Evaluation**: Against established ReqBERT baselines
- **Practical Implementation**: Production-ready workflow with complete evaluation framework
- **Research Contribution**: Bridge between academic research and industrial application

### For PhD Thesis Chapter:
- **End-to-end Pipeline**: From requirement text to semantic features
- **Performance Analysis**: Detailed comparison with state-of-the-art
- **Industrial Relevance**: Scalable solution for requirements engineering
- **Future Work Foundation**: Basis for Digital Twin integration

## 🛠️ Development and Customization

### Extending the Workflow
1. **Custom Classifiers**: Add new classification models
2. **Additional Subclasses**: Extend NFR taxonomy
3. **Domain Adaptation**: Customize for specific engineering domains
4. **Integration APIs**: Connect with existing PLM/ALM systems

### Performance Optimization
1. **Prompt Engineering**: Refine classification prompts
2. **Few-shot Examples**: Curate domain-specific examples
3. **Ensemble Methods**: Combine multiple approaches
4. **Caching**: Implement result caching for efficiency

## 📋 Requirements

### Python Dependencies
- `langchain>=0.1.0` - LLM integration framework
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Evaluation metrics
- `datasets>=2.0.0` - HuggingFace dataset loading
- `matplotlib>=3.5.0` - Visualization

### API Access
- **Anthropic API Key** - Required for Claude model access
- **HuggingFace Account** - Optional for dataset access

## 📞 Support and Contact

**Author:** Alessandro Stefanone  
**Institution:** Politecnico di Milano  
**Program:** PhD in Mechanical Engineering  
**Research Focus:** AI applications to industrial engineering, digital twins, PLM

**Websites:**
- Personal: https://alessandrostefanone.github.io/
- LinkedIn: https://www.linkedin.com/in/alessandro-stefanone-784a19134/

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@software{stefanone2024hierarchical,
  title={Hierarchical Requirements Classification AI Workflow},
  author={Stefanone, Alessandro},
  year={2024},
  institution={Politecnico di Milano},
  url={https://github.com/alessandrostefanone-polimi/fr-nfr-classification}
}
```

## 📜 License

This work is part of ongoing PhD research at Politecnico di Milano. Please contact the author for usage permissions and collaboration opportunities.

---

*This implementation demonstrates the successful integration of state-of-the-art NLP techniques with practical requirements engineering workflows, providing a foundation for advanced semantic feature extraction in technical document analysis.*
