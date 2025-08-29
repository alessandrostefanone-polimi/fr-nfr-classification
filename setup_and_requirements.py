#!/usr/bin/env python3
"""
Setup script for FR/NFR Classification Pipeline (Windows Compatible)

This script sets up the environment and dependencies for the 
Functional/Non-Functional Requirements classification pipeline.

Usage:
    python setup_and_requirements.py
"""

import subprocess
import sys
import os
from pathlib import Path

def create_requirements_txt():
    """Create requirements.txt with all necessary dependencies"""
    requirements = """
# Core LangChain and LLM dependencies
langchain>=0.1.0
langchain-community>=0.0.10
langchain-anthropic>=0.1.0
anthropic>=0.18.0

# Data processing and ML
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Output parsing and validation
pydantic>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Progress bars and utilities
tqdm>=4.65.0
requests>=2.31.0

# Optional: For local LLM support via Ollama
# (Ollama needs to be installed separately)

# Development and testing
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
jupyter>=1.0.0
"""
    
    with open("requirements.txt", "w", encoding='utf-8') as f:
        f.write(requirements.strip())
    
    print("✅ Created requirements.txt")

def create_env_template():
    """Create environment template file"""
    env_template = """
# Environment Variables for FR/NFR Classification Pipeline

# Anthropic Claude API (required for cloud-based LLM)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: OpenAI API (if extending to GPT models)
# OPENAI_API_KEY=your-openai-api-key-here

# Optional: Ollama settings (for local LLM)
OLLAMA_BASE_URL=http://localhost:11434

# Output directory for results
OUTPUT_DIR=./evaluation_results

# Logging level
LOG_LEVEL=INFO
"""
    
    with open(".env.template", "w", encoding='utf-8') as f:
        f.write(env_template.strip())
    
    print("✅ Created .env.template")
    print("   📝 Copy to .env and fill in your API keys")

def install_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "evaluation_results",
        "data",
        "logs",
        "output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_example_usage():
    """Create example usage script"""
    example_script = '''#!/usr/bin/env python3
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
    print("\\n1. Testing with sample requirements...")
    
    # Initialize classifier
    classifier = RequirementClassifier(
        model_type="claude",
        model_name="claude-3-sonnet-20240229",
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
        print(f"\\n--- Requirement {i} ---")
        print(f"Text: {req}")
        
        result = classifier.classify_requirement(req)
        print(f"Classification: {result.predicted_label}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Key Indicators: {', '.join(result.key_indicators)}")
    
    # Example 2: Batch classification with evaluation
    print("\\n\\n2. Batch classification with evaluation...")
    
    # Load test data
    test_reqs, test_labels = DatasetLoader.create_test_samples()
    
    # Classify batch
    results = classifier.classify_batch(test_reqs, test_labels)
    
    # Evaluate performance
    metrics = classifier.evaluate_performance(results)
    
    print(f"\\nPerformance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    
    # Save results
    classifier.save_results(results, metrics, "output/example_results/")
    print("\\nResults saved to output/example_results/")

if __name__ == "__main__":
    main()
'''
    
    with open("example_usage.py", "w", encoding='utf-8') as f:
        f.write(example_script)
    
    print("✅ Created example_usage.py")

def create_comprehensive_evaluation_script():
    """Create script to run comprehensive evaluation"""
    eval_script = '''#!/usr/bin/env python3
"""
Run comprehensive evaluation comparing LLMs with BERT baselines

This script runs the full evaluation pipeline comparing multiple LLM
configurations against BERT baselines from the literature.

Usage:
    python run_evaluation.py [--full-dataset] [--sample-size N]
"""

import argparse
from comprehensive_evaluation import main as run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FR/NFR Classification Evaluation')
    parser.add_argument('--full-dataset', action='store_true',
                       help='Use full dataset (956 samples)')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Sample size for evaluation (default: 50)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("Starting Comprehensive FR/NFR Classification Evaluation")
    print(f"Dataset: {'Full dataset' if args.full_dataset else f'{args.sample_size} samples'}")
    print(f"Output: {args.output_dir}")
    
    # Run evaluation
    run_evaluation()
'''
    
    with open("run_evaluation.py", "w", encoding='utf-8') as f:
        f.write(eval_script)
    
    print("✅ Created run_evaluation.py")

def create_readme():
    """Create README with documentation"""
    readme_content = '''# FR/NFR Classification Pipeline

A comprehensive pipeline for classifying software requirements into Functional Requirements (FR) and Non-Functional Requirements (NFR) using Large Language Models (LLMs) and comparing performance with BERT baselines.

## Objective

This project implements an end-to-end semantic feature extraction workflow for technical requirements, comparing state-of-the-art LLMs with transformer-based models from academic literature.

## Sources and Citations

### Primary Literature:
1. **Lim, S. C. (2022).** "A Case for Pre-trained Language Models in Systems Engineering." *MIT System Design and Management Program.*
2. **ISO/IEC 25010:2023** - Systems and software Quality Requirements and Evaluation (SQuaRE)
3. **IEEE Standards** for Requirements Engineering

### Supporting Research:
- Ferrari, A., et al. (2021). "NLP for Requirements Engineering: Tasks, Techniques, Tools, and Technologies"
- Zhao, L., et al. (2021). "Natural Language Processing for Requirements Engineering: A Systematic Mapping Study"
- Chen, L., Babar, M. A., & Nuseibeh, B. (2013). "Characterizing Architecturally Significant Requirements"

### Dataset:
- **HuggingFace Dataset:** `limsc/fr-nfr-classification` (956 labeled requirements)

## Quick Start

### 1. Setup Environment

```bash
# Clone/download the project files
# Run setup script
python setup_and_requirements.py

# Copy environment template and add your API keys
cp .env.template .env
# Edit .env with your Anthropic API key
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Example

```bash
# Set your API key
set ANTHROPIC_API_KEY=your-api-key-here

# Run example classification
python example_usage.py
```

### 4. Run Comprehensive Evaluation

```bash
# Quick evaluation (50 samples)
python run_evaluation.py --sample-size 50

# Full evaluation (956 samples) - takes longer
python run_evaluation.py --full-dataset
```

## Architecture

### Core Components:

1. **`requirements_literature_review.py`**
   - Classification rules extracted from IEEE standards and ISO/IEC 25010
   - Linguistic patterns from requirements engineering literature
   - Comprehensive rule base for FR/NFR distinction

2. **`langchain_fr_nfr_pipeline.py`**
   - LangChain-based pipeline supporting multiple LLMs
   - Supports both cloud (Anthropic Claude) and local (Ollama) models
   - Structured output parsing with Pydantic models
   - Comprehensive evaluation framework

3. **`comprehensive_evaluation.py`**
   - Full evaluation pipeline comparing LLM vs BERT performance
   - Statistical analysis and visualization generation
   - Comparison with published BERT baselines

### Classification Rules:

**Functional Requirements (FR):**
- Define WHAT the system must do
- Action verbs: calculate, process, display, store
- Patterns: "shall perform", "must execute"
- Example: "The system shall calculate monthly interest"

**Non-Functional Requirements (NFR):**
- Define HOW WELL the system performs
- Based on ISO/IEC 25010:2023 quality characteristics
- Quality attributes: performance, security, usability
- Patterns: "within X seconds", "99.9% uptime"
- Example: "System shall respond within 2 seconds"

## Configuration Options

### LLM Models Supported:

**Cloud Models (via Anthropic):**
- Claude-3 Sonnet
- Claude-3 Haiku
- Claude-3 Opus

**Local Models (via Ollama):**
- Llama 3.1 (8B, 70B)
- Mistral 7B
- Qwen2 7B
- Any Ollama-supported model

### Prompting Strategies:
- **Few-shot learning**: Examples included in prompt
- **Zero-shot**: Direct classification without examples
- **Rule-based**: Comprehensive classification rules from literature

## Performance Baselines

### BERT Baselines (from Lim 2022):
- BERT Base: 0.838 F1-Score
- RoBERTa Base: 0.826 F1-Score
- SciBERT: 0.850 F1-Score
- ReqBERT (TAPT): 0.841 F1-Score

### Evaluation Metrics:
- F1-Score (weighted average)
- Precision and Recall
- Accuracy
- Detailed classification report
- Confusion matrices

## Output Structure

```
evaluation_results/
├── comprehensive_evaluation_report.json  # Detailed results
├── evaluation_summary.md                 # Human-readable summary
├── performance_comparison.png             # Performance visualization
├── llm_metrics_heatmap.png               # Metrics heatmap
└── [model_config_name]/
    ├── classification_results_*.csv       # Detailed classifications
    └── evaluation_metrics_*.json          # Model-specific metrics
```

## Key Features

### 1. Comprehensive Rule Base
- Extracted from IEEE standards and ISO/IEC 25010:2023
- Covers all major quality characteristics
- Linguistically-grounded classification patterns

### 2. Multi-Model Support
- Cloud and local LLM support
- Easy model switching and comparison
- Consistent evaluation framework

### 3. Explainable Results
- Detailed reasoning for each classification
- Key indicator identification
- Confidence scoring

### 4. Academic Rigor
- Proper citation of all sources
- Comparison with published baselines
- Reproducible methodology

## Usage Examples

### Basic Classification:

```python
from langchain_fr_nfr_pipeline import RequirementClassifier

classifier = RequirementClassifier(
    model_type="claude",
    model_name="claude-3-sonnet-20240229",
    use_few_shot=True
)

result = classifier.classify_requirement(
    "The system shall respond within 2 seconds"
)
print(f"Classification: {result.predicted_label}")
print(f"Reasoning: {result.reasoning}")
```

### Batch Evaluation:

```python
requirements = ["req1", "req2", "req3"]
labels = ["FR", "NFR", "FR"]

results = classifier.classify_batch(requirements, labels)
metrics = classifier.evaluate_performance(results)
print(f"F1-Score: {metrics['f1_score']:.3f}")
```

## Troubleshooting

### Common Issues:

1. **API Key Issues**: Ensure ANTHROPIC_API_KEY is set correctly
2. **Ollama Connection**: Verify Ollama server is running on localhost:11434
3. **Memory Issues**: Reduce sample size for evaluation
4. **Rate Limits**: Add delays between API calls if needed
5. **Unicode Issues**: Files are saved with UTF-8 encoding for Windows compatibility

### Support:
- Check logs in `logs/` directory
- Review error messages in console output
- Verify all dependencies are installed correctly

---

*This pipeline demonstrates the application of state-of-the-art NLP techniques to Requirements Engineering, bridging academic research with practical implementation.*
'''
    
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created README.md")

def main():
    """Main setup function"""
    print("Setting up FR/NFR Classification Pipeline")
    print("=" * 50)
    
    # Create requirements and environment files
    create_requirements_txt()
    create_env_template()
    
    # Setup directories
    setup_directories()
    
    # Create example and evaluation scripts
    create_example_usage()
    create_comprehensive_evaluation_script()
    
    # Create documentation
    create_readme()
    
    # Install dependencies
    if input("\nInstall Python dependencies now? (y/N): ").lower().startswith('y'):
        install_dependencies()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Run: python example_usage.py (for quick test)")
    print("3. Run: python run_evaluation.py (for full evaluation)")
    print("4. Check README.md for detailed documentation")
    print("\nReady to classify requirements!")

if __name__ == "__main__":
    main()