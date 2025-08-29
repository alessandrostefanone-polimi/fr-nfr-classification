"""
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
    print("\nTesting sample data...")
    
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
    print("\nTesting benchmark data...")
    
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
        print(f"\nTesting {test_name}:")
        if test_func():
            passed += 1
            print(f"SUCCESS: {test_name} passed")
        else:
            print(f"ERROR: {test_name} failed")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All tests passed! The workflow is ready to use.")
        print("\nNext steps:")
        print("1. Set your ANTHROPIC_API_KEY in .env file")
        print("2. Run: python demo.py")
        print("3. Run: python benchmark_evaluation.py")
    else:
        print(f"\nWARNING: {total - passed} tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
