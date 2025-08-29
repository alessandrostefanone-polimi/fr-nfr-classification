#!/usr/bin/env python3
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
        print("\n✅ All tests passed! Ready for full evaluation.")
    else:
        print("\n❌ Tests failed. Please check your setup.")
        sys.exit(1)
