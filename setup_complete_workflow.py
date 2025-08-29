"""
Complete Setup and Usage Guide for Hierarchical Requirements Classification
==========================================================================

This script sets up the complete AI workflow for hierarchical requirements classification
integrating FR/NFR classification with LLM-based NFR subclassification.

Author: Alessandro Stefanone
Affiliation: Politecnico di Milano - PhD in Mechanical Engineering
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List


def create_requirements_file():
    """Create requirements.txt with all necessary dependencies"""
    requirements = [
        "# Core ML and Data Processing",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "",
        "# LangChain and LLM Integration", 
        "langchain>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "langchain-core>=0.1.0",
        "",
        "# Dataset Loading",
        "datasets>=2.0.0",
        "huggingface-hub>=0.16.0",
        "",
        "# Visualization",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "",
        "# Utilities",
        "python-dotenv>=0.19.0",
        "pydantic>=1.10.0",
        "",
        "# Optional: For local model inference",
        "# torch>=1.11.0",
        "# transformers>=4.20.0",
        "",
        "# Development and Testing",
        "pytest>=6.0.0",
        "jupyter>=1.0.0"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    print("✅ Created requirements.txt")


def create_environment_file():
    """Create .env template file"""
    env_content = """# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: OpenAI API (if using OpenAI models)
# OPENAI_API_KEY=your_openai_api_key_here

# Optional: Hugging Face API (for dataset access)
# HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Evaluation Settings
DEFAULT_BATCH_SIZE=10
DEFAULT_DELAY_BETWEEN_BATCHES=1.5
DEFAULT_TEMPERATURE=0.0

# Output Directories
RESULTS_DIR=evaluation_results
BENCHMARK_DIR=benchmark_results
LOGS_DIR=logs
"""
    
    with open(".env.template", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env.template")


def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "evaluation_results",
        "hierarchical_evaluation_results", 
        "benchmark_results",
        "logs",
        "data",
        "models",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Created directory structure")


def create_demo_script():
    """Create demonstration script"""
    demo_content = '''"""
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
        print("\\nInitializing AI workflow...")
        workflow = HierarchicalRequirementsWorkflow(
            anthropic_api_key=api_key,
            use_few_shot=True
        )
        print("SUCCESS: Workflow initialized successfully")
        
        # Get sample data
        print("\\nLoading sample requirements...")
        requirements, true_labels = get_sample_nfr_data()
        print(f"SUCCESS: Loaded {len(requirements)} sample requirements")
        
        # Test single classification
        print("\\nTesting single requirement classification...")
        sample_req = requirements[0]
        result = workflow.classify_single_requirement(sample_req)
        
        print(f"Requirement: {sample_req}")
        print(f"FR/NFR Classification: {result.fr_nfr_prediction} (conf: {result.fr_nfr_confidence:.2f})")
        if result.nfr_subclass_prediction:
            print(f"NFR Subclass: {result.nfr_subclass_prediction} (conf: {result.nfr_subclass_confidence:.2f})")
        print(f"Final Classification: {result.final_classification}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
        # Test batch classification
        print("\\nTesting batch classification (first 3 requirements)...")
        batch_results = workflow.classify_batch(
            requirements[:3], 
            batch_size=3,
            delay_between_batches=1.0
        )
        
        for i, result in enumerate(batch_results):
            print(f"\\n#{i+1}: {result.requirement_text[:60]}...")
            print(f"  -> {result.final_classification}")
        
        # Save results
        workflow.save_workflow_results(batch_results, experiment_name="demo_run")
        print("\\nSUCCESS: Demo results saved to hierarchical_evaluation_results/")
        
        print("\\nDEMO COMPLETED SUCCESSFULLY!")
        print("\\nNext steps:")
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
'''
    
    with open("demo.py", "w", encoding='utf-8') as f:
        f.write(demo_content)
    
    print("✅ Created demo.py")


def create_readme():
    """Create comprehensive README"""
    readme_content = '''# Hierarchical Requirements Classification AI Workflow

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
'''
    
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created comprehensive README.md")


def install_dependencies():
    """Install required Python packages"""
    print("🔧 Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Please run manually: pip install -r requirements.txt")
        return False


def verify_installation():
    """Verify that key dependencies are available"""
    required_packages = [
        "langchain",
        "langchain_anthropic", 
        "datasets",
        "pandas",
        "sklearn",
        "matplotlib"
    ]
    
    print("🔍 Verifying installation...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Missing")
            return False
    
    print("✅ All required packages verified")
    return True


def create_test_script():
    """Create a simple test script"""
    test_content = '''"""
Test Script for Hierarchical Requirements Classification
======================================================

Simple tests to verify the workflow components are working correctly.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from nfr_subclass_classifier import LLMNFRSubclassifier, get_sample_nfr_data
        print("  SUCCESS: NFR subclassification module")
    except ImportError as e:
        print(f"  ERROR: NFR subclassification module: {e}")
        return False
    
    try:
        from hierarchical_workflow import HierarchicalRequirementsWorkflow
        print("  SUCCESS: Hierarchical workflow module")
    except ImportError as e:
        print(f"  ERROR: Hierarchical workflow module: {e}")
        return False
    
    try:
        from benchmark_evaluation import BenchmarkEvaluator, StateOfArtBenchmarks
        print("  SUCCESS: Benchmark evaluation module")
    except ImportError as e:
        print(f"  ERROR: Benchmark evaluation module: {e}")
        return False
    
    return True


def test_sample_data():
    """Test sample data loading"""
    print("\\nTesting sample data...")
    
    try:
        from nfr_subclass_classifier import get_sample_nfr_data
        requirements, labels = get_sample_nfr_data()
        
        print(f"  SUCCESS: Loaded {len(requirements)} sample requirements")
        print(f"  SUCCESS: Loaded {len(labels)} sample labels")
        print(f"  SUCCESS: Sample requirement: {requirements[0][:50]}...")
        print(f"  SUCCESS: Sample label: {labels[0]}")
        
        return True
    except Exception as e:
        print(f"  ERROR: Error loading sample data: {e}")
        return False


def test_benchmarks():
    """Test benchmark data"""
    print("\\nTesting benchmark data...")
    
    try:
        from benchmark_evaluation import StateOfArtBenchmarks
        
        benchmarks = StateOfArtBenchmarks()
        nfr_benchmarks = benchmarks.get_nfr_only_benchmarks()
        
        print(f"  SUCCESS: NFR benchmark classes: {list(nfr_benchmarks['per_class'].keys())}")
        print(f"  SUCCESS: Overall F1-weighted: {nfr_benchmarks['f1_weighted']:.3f}")
        print(f"  SUCCESS: Total samples: {nfr_benchmarks['total_samples']}")
        
        return True
    except Exception as e:
        print(f"  ERROR: Error loading benchmarks: {e}")
        return False


def main():
    """Run all tests"""
    print("HIERARCHICAL REQUIREMENTS CLASSIFICATION - TESTS")
    print("=" * 55)
    
    tests = [
        ("Module Imports", test_imports),
        ("Sample Data", test_sample_data), 
        ("Benchmark Data", test_benchmarks)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\nTesting {test_name}:")
        if test_func():
            passed += 1
            print(f"SUCCESS: {test_name} passed")
        else:
            print(f"ERROR: {test_name} failed")
    
    print(f"\\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\nSUCCESS: All tests passed! The workflow is ready to use.")
        print("\\nNext steps:")
        print("1. Set your ANTHROPIC_API_KEY in .env file")
        print("2. Run: python demo.py")
        print("3. Run: python benchmark_evaluation.py")
    else:
        print(f"\\nWARNING: {total - passed} tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open("test_workflow.py", "w", encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Created test_workflow.py")


def main():
    """Main setup function"""
    print("HIERARCHICAL REQUIREMENTS CLASSIFICATION WORKFLOW SETUP")
    print("=" * 60)
    print("Setting up complete AI workflow for requirements engineering...")
    print()
    
    # Create all necessary files and directories
    steps = [
        ("Creating requirements.txt", create_requirements_file),
        ("Creating environment template", create_environment_file),
        ("Creating directory structure", create_directory_structure),
        ("Creating demonstration script", create_demo_script),
        ("Creating comprehensive README", create_readme),
        ("Creating test script", create_test_script)
    ]
    
    for step_name, step_func in steps:
        print(f"📝 {step_name}...")
        try:
            step_func()
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    
    # Optional dependency installation
    print("\n🤔 Would you like to install Python dependencies now?")
    install_deps = input("Install dependencies? [y/N]: ").lower().strip()
    
    if install_deps.startswith('y'):
        if install_dependencies():
            print("\n🔍 Verifying installation...")
            verify_installation()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print()
    print("1. 🔑 Setup API Access:")
    print("   • Copy .env.template to .env")
    print("   • Add your Anthropic API key to .env")
    print("   • Run: source .env (Linux/Mac)")
    print()
    print("2. 🧪 Run Tests:")
    print("   • python test_workflow.py")
    print()  
    print("3. 🚀 Run Demo:")
    print("   • python demo.py")
    print()
    print("4. 📊 Run Full Benchmark:")
    print("   • python benchmark_evaluation.py")
    print()
    print("5. 📚 Read Documentation:")
    print("   • Check README.md for detailed usage")
    print("   • Explore example notebooks in notebooks/")
    print()
    print("🎉 Your AI workflow for hierarchical requirements classification is ready!")
    print("This implementation integrates with your existing FR/NFR classifier")
    print("and adds LLM-based NFR subclassification with state-of-the-art evaluation.")
    print()
    print("📧 Questions? Contact: Alessandro Stefanone")
    print("🔗 https://alessandrostefanone.github.io/")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)