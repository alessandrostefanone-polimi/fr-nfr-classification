#!/usr/bin/env python3
"""
Install and Test Script for FR/NFR Classification Pipeline

This script handles installation and testing with proper error handling.
Run this first to set up everything correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_packages():
    """Install packages with compatible versions"""
    print("📦 Installing compatible packages...")
    
    packages = [
        "langchain",
        "langchain-community", 
        "langchain-anthropic>=0.1.15",
        "anthropic>=0.28.0",
        "datasets>=2.14.0",
        "pandas>=2.0.0", 
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to install {package}: {e}")
    
    print("✅ Installation completed!")

def check_imports():
    """Check if all required imports work"""
    print("\n🔍 Checking imports...")
    
    try:
        import langchain
        print(f"✅ langchain: {langchain.__version__}")
    except ImportError as e:
        print(f"❌ langchain import failed: {e}")
        return False
    
    try:
        from langchain_anthropic import ChatAnthropic
        print("✅ langchain_anthropic: OK")
    except ImportError as e:
        print(f"❌ langchain_anthropic import failed: {e}")
        return False
    
    try:
        import anthropic
        print(f"✅ anthropic: {anthropic.__version__}")
    except ImportError as e:
        print(f"❌ anthropic import failed: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✅ datasets: OK")
    except ImportError as e:
        print(f"❌ datasets import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ pandas, numpy, sklearn: OK")
    except ImportError as e:
        print(f"❌ Data libraries import failed: {e}")
        return False
    
    return True

def test_anthropic_connection():
    """Test if Anthropic API key works"""
    print("\n🔑 Testing Anthropic API connection...")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        print("⚠️  ANTHROPIC_API_KEY not set. Please set it to test the connection.")
        return False
    
    try:
        from langchain_anthropic import ChatAnthropic
        
        llm = ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key,
            max_tokens=100
        )
        
        response = llm.invoke("Say 'Hello, API test successful!'")
        print(f"✅ API test successful: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def create_simple_test():
    """Create a simple test script"""
    test_script = '''#!/usr/bin/env python3
"""
Simple test for FR/NFR Classification
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_classification():
    """Test the classification pipeline"""
    try:
        # Check API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-anthropic-api-key-here":
            print("❌ Please set ANTHROPIC_API_KEY environment variable")
            return False
        
        # Import the pipeline
        from fixed_langchain_pipeline import RequirementClassifier
        
        # Create classifier
        classifier = RequirementClassifier(
            model_type="claude",
            model_name="claude-3-5-sonnet-20241022",
            use_few_shot=True,
            temperature=0.1
        )
        
        # Test requirements
        test_cases = [
            ("The system shall calculate monthly interest rates.", "FR"),
            ("The system shall respond within 2 seconds.", "NFR"),
            ("Users can create new accounts.", "FR"),
            ("The application must maintain 99% uptime.", "NFR")
        ]
        
        print("🧪 Testing classification...")
        correct = 0
        total = len(test_cases)
        
        for req, expected in test_cases:
            result = classifier.classify_requirement(req)
            is_correct = result.predicted_label == expected
            correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {req}")
            print(f"   Predicted: {result.predicted_label} | Expected: {expected}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Reasoning: {result.reasoning[:80]}...")
            print()
        
        accuracy = correct / total
        print(f"📊 Test Results: {correct}/{total} correct ({accuracy:.1%})")
        
        if accuracy >= 0.75:
            print("🎉 Test PASSED! Pipeline is working correctly.")
            return True
        else:
            print("⚠️  Test results below 75% accuracy. Check the pipeline.")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Simple FR/NFR Classification Test")
    print("=" * 40)
    
    success = test_classification()
    if success:
        print("\\n✅ All tests passed! Ready for full evaluation.")
    else:
        print("\\n❌ Tests failed. Please check your setup.")
        sys.exit(1)
'''
    
    with open("simple_test.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ Created simple_test.py")

def main():
    """Main setup and test function"""
    print("🚀 FR/NFR Classification - Install & Test")
    print("=" * 50)
    
    # Install packages
    install_packages()
    
    # Check imports
    if not check_imports():
        print("\n❌ Import check failed. Please fix package installations.")
        sys.exit(1)
    
    # Test API connection
    test_anthropic_connection()
    
    # Create simple test
    create_simple_test()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed!")
    print("\n📝 Next steps:")
    print("1. Set your API key:")
    print("   set ANTHROPIC_API_KEY=your-api-key-here")
    print("2. Run simple test:")
    print("   python simple_test.py")
    print("3. If test passes, run evaluation:")
    print("   python run_evaluation.py --sample-size 5")
    
    print("\n🎯 Ready to classify requirements!")

if __name__ == "__main__":
    main()