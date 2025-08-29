"""
Enhanced Benchmark Evaluation with Sample Size Control
======================================================

Run benchmark evaluation with customizable sample sizes for prompt optimization.

Usage examples:
  python benchmark_evaluation.py --samples 50    # Test with 50 samples
  python benchmark_evaluation.py --samples 10    # Quick test with 10 samples  
  python benchmark_evaluation.py --full          # Full dataset evaluation
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from hierarchical_workflow import HierarchicalRequirementsWorkflow
from benchmark_evaluation import BenchmarkEvaluator
from nfr_subclass_classifier import get_sample_nfr_data


def run_benchmark_with_samples(sample_size: int = 50, save_results: bool = True):
    """
    Run benchmark evaluation with specified sample size
    
    Args:
        sample_size: Number of samples to evaluate
        save_results: Whether to save detailed results
        
    Returns:
        Benchmark results dictionary
    """
    print(f"BENCHMARK EVALUATION - {sample_size} SAMPLES")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("Please set your API key in the .env file")
        return None
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        
        workflow = HierarchicalRequirementsWorkflow(
            anthropic_api_key=api_key,
            use_few_shot=True,
            temperature=0.0
        )
        
        evaluator = BenchmarkEvaluator(
            output_dir=f"benchmark_results_sample_{sample_size}"
        )
        
        print("✅ Components initialized successfully")
        
        # Run benchmark with specified sample size
        print(f"\n🚀 Running benchmark evaluation on {sample_size} samples...")
        print("⏱️  This may take several minutes depending on sample size...")
        
        results = evaluator.run_comprehensive_benchmark(
            workflow=workflow,
            dataset_size=sample_size,
            save_results=save_results
        )
        
        if 'error' in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return None
        
        # Display results
        print_benchmark_summary(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_benchmark_summary(results: dict):
    """Print a summary of benchmark results"""
    print("\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    
    dataset_info = results['dataset_info']
    overall_comp = results['comparison']['overall_comparison']
    perf_summary = results['comparison']['performance_summary']
    
    print(f"📈 Dataset Info:")
    print(f"   Total Samples: {dataset_info['total_samples']}")
    print(f"   NFR Samples Evaluated: {dataset_info['evaluation_samples']}")
    
    print(f"\n🎯 Overall Performance:")
    print(f"   LLM F1-Score (Weighted): {overall_comp['f1_weighted']['llm']:.3f}")
    print(f"   SOTA F1-Score (Weighted): {overall_comp['f1_weighted']['sota']:.3f}")
    print(f"   Improvement: {overall_comp['f1_weighted']['improvement']:+.1f}%")
    
    print(f"\n📋 Per-Class Results:")
    per_class_comp = results['comparison']['per_class_comparison']
    for class_name, comp_data in per_class_comp.items():
        status = "✅" if comp_data['improvement'] > 0 else "❌" if comp_data['improvement'] < -2 else "⚠️"
        print(f"   {status} {class_name} ({comp_data['class_full_name']}): "
              f"LLM={comp_data['llm_f1']:.3f}, SOTA={comp_data['sota_f1']:.3f}, "
              f"Δ={comp_data['improvement']:+.1f}%")
    
    print(f"\n🏆 Performance Summary:")
    print(f"   Classes Improved: {perf_summary['classes_improved']}/4")
    print(f"   Average Improvement: {perf_summary['average_improvement']:+.1f}%")
    print(f"   Competitive Performance: {'✅ Yes' if perf_summary['overall_competitive'] else '❌ No'}")
    
    # Recommendations for prompt optimization
    print(f"\n💡 PROMPT OPTIMIZATION INSIGHTS:")
    
    # Identify weak classes for prompt improvement
    weak_classes = [name for name, data in per_class_comp.items() 
                   if data['improvement'] < -2]
    strong_classes = [name for name, data in per_class_comp.items() 
                     if data['improvement'] > 5]
    
    if weak_classes:
        print(f"   🔴 Focus prompt improvement on: {', '.join(weak_classes)}")
        for class_name in weak_classes:
            comp_data = per_class_comp[class_name]
            print(f"      - {class_name}: Current F1={comp_data['llm_f1']:.3f}, Target={comp_data['sota_f1']:.3f}")
    
    if strong_classes:
        print(f"   🟢 Strong performance in: {', '.join(strong_classes)}")
        print(f"      - Use these as examples for other classes")


def interactive_prompt_optimization():
    """Interactive mode for prompt optimization"""
    print("\n🔬 INTERACTIVE PROMPT OPTIMIZATION")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Test with 10 samples (quick)")
        print("2. Test with 25 samples (medium)")
        print("3. Test with 50 samples (thorough)")
        print("4. Custom sample size")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            run_benchmark_with_samples(10)
        elif choice == "2":
            run_benchmark_with_samples(25)
        elif choice == "3":
            run_benchmark_with_samples(50)
        elif choice == "4":
            try:
                custom_size = int(input("Enter sample size: "))
                run_benchmark_with_samples(custom_size)
            except ValueError:
                print("Invalid number. Please try again.")
        elif choice == "5":
            break
        else:
            print("Invalid choice. Please select 1-5.")
        
        continue_opt = input("\nRun another test? (y/N): ").lower().strip()
        if not continue_opt.startswith('y'):
            break


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Benchmark NFR Subclassification with customizable sample sizes"
    )
    parser.add_argument(
        '--samples', '-s', 
        type=int, 
        default=50,
        help='Number of samples to evaluate (default: 50)'
    )
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Run full dataset evaluation (overrides --samples)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode for prompt optimization'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving detailed results (faster)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if args.interactive:
        interactive_prompt_optimization()
    else:
        sample_size = None if args.full else args.samples
        save_results = not args.no_save
        
        results = run_benchmark_with_samples(
            sample_size=sample_size or 1000,  # Large number for full dataset
            save_results=save_results
        )
        
        if results:
            print(f"\n💾 Results saved to benchmark_results_sample_{sample_size or 'full'}/")
            print("\n🎉 Benchmark evaluation completed!")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()