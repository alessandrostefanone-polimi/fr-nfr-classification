"""
Fixed Pipeline Runner with Proper Dataset Loading
================================================

This script runs the complete hierarchical requirements classification pipeline
with proper dataset handling and label mapping.
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def load_environment():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("⚠️  python-dotenv not available, using system environment")

def check_api_key():
    """Check if API key is available"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        print("\nPlease set your API key:")
        print("1. Copy .env.template to .env")
        print("2. Edit .env and add: ANTHROPIC_API_KEY=your_key_here")
        return None
    
    print(f"✅ API key found: ***{api_key[-4:]}")
    return api_key

def load_proper_test_data():
    """Load proper test data with real requirements and correct labels"""
    
    print("📂 Creating curated test dataset with real requirements...")
    
    # Real requirements with proper classifications
    requirements = [
        # Performance (PE) requirements
        "The system shall respond to user queries within 2 seconds under normal load conditions.",
        "The application must handle up to 1000 concurrent users without performance degradation.",
        "Database queries shall execute in less than 500 milliseconds for 95% of requests.",
        "The system throughput shall be at least 100 transactions per second.",
        
        # Security (SE) requirements
        "All user passwords must be encrypted using AES-256 encryption standard.",
        "The system shall implement two-factor authentication for all admin users.",
        "User sessions must expire after 30 minutes of inactivity for security.",
        "All data transmissions shall use HTTPS with TLS 1.3 or higher encryption.",
        
        # Usability (US) requirements
        "The user interface shall be intuitive for users with minimal training required.",
        "The system shall provide context-sensitive help for all major functions.",
        "Error messages shall be displayed in plain language understandable to end users.",
        "The application shall be accessible to users with visual impairments following WCAG guidelines.",
        
        # Operational (O) requirements
        "The system shall provide comprehensive logging of all user actions for audit purposes.",
        "System backups shall be performed automatically every 24 hours without user intervention.",
        "The application shall send email notifications for critical system events to administrators.",
        "System administrators shall receive alerts when disk usage exceeds 80% capacity.",
        
        # Functional (FR) requirements for comparison
        "Users shall be able to create new customer records in the database.",
        "The system shall calculate monthly interest rates based on account balances.",
        "Customers shall be able to transfer funds between their own accounts.",
        "The application shall generate monthly statements for all active accounts automatically."
    ]
    
    # Corresponding correct labels
    true_labels = [
        # Performance
        'PE', 'PE', 'PE', 'PE',
        # Security
        'SE', 'SE', 'SE', 'SE',
        # Usability
        'US', 'US', 'US', 'US',
        # Operational
        'O', 'O', 'O', 'O',
        # Functional
        'FR', 'FR', 'FR', 'FR'
    ]
    
    print(f"✅ Created test dataset with {len(requirements)} requirements")
    
    # Show label distribution
    label_counts = {}
    for label in true_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"📊 Label distribution: {label_counts}")
    
    return requirements, true_labels

def run_pipeline_evaluation(sample_size=20):
    """
    Run complete pipeline evaluation with proper test data
    
    Args:
        sample_size: Number of samples to evaluate
    """
    print(f"COMPLETE HIERARCHICAL PIPELINE EVALUATION")
    print("=" * 50)
    print(f"Sample Size: {sample_size}")
    print(f"Expected Duration: ~{sample_size * 0.3:.0f} minutes\n")
    
    try:
        # Import components
        print("📦 Importing pipeline components...")
        from langchain_fr_nfr_pipeline import RequirementClassifier
        from nfr_subclass_classifier import LLMNFRSubclassifier
        print("✅ All components imported successfully\n")
        
        # Initialize FR/NFR classifier
        print("🔧 Initializing FR/NFR classifier...")
        fr_nfr_classifier = RequirementClassifier(
            model_type="claude",
            model_name="claude-3-5-haiku-20241022",
            use_few_shot=True
        )
        print("✅ FR/NFR classifier ready")
        
        # Initialize NFR subclassifier
        print("🔧 Initializing NFR subclassifier...")
        api_key = os.getenv('ANTHROPIC_API_KEY')
        nfr_subclassifier = LLMNFRSubclassifier(
            model_type="claude",
            model_name="claude-3-5-haiku-20241022",
            api_key=api_key,
            use_few_shot=True,
            temperature=0.0
        )
        print("✅ NFR subclassifier ready\n")
        
        # Load proper test data
        requirements, true_labels = load_proper_test_data()
        
        # Limit to sample size
        if sample_size < len(requirements):
            requirements = requirements[:sample_size]
            true_labels = true_labels[:sample_size]
            print(f"📊 Limited to {len(requirements)} samples for evaluation\n")
        
        print(f"🚀 Starting pipeline evaluation...")
        
        # Run complete pipeline
        pipeline_results = []
        correct_predictions = 0
        fr_count = 0
        nfr_count = 0
        nfr_subclass_counts = {'O': 0, 'PE': 0, 'SE': 0, 'US': 0, 'UNKNOWN': 0}
        processing_times = []
        
        for i, (requirement, true_label) in enumerate(zip(requirements, true_labels)):
            start_time = time.time()
            
            print(f"\n--- Processing {i+1}/{len(requirements)} ---")
            print(f"Requirement: {requirement[:80]}...")
            print(f"True Label: {true_label}")
            
            # Step 1: FR/NFR Classification
            fr_nfr_result = fr_nfr_classifier.classify_requirement(requirement)
            fr_nfr_pred = fr_nfr_result.predicted_label
            fr_nfr_conf = fr_nfr_result.confidence
            
            print(f"FR/NFR: {fr_nfr_pred} (confidence: {fr_nfr_conf:.2f})")
            
            # Count FR/NFR
            if fr_nfr_pred == "FR":
                fr_count += 1
            else:
                nfr_count += 1
            
            final_prediction = fr_nfr_pred
            nfr_subclass_pred = None
            nfr_subclass_conf = None
            
            # Step 2: NFR Subclassification (if NFR)
            if fr_nfr_pred == "NFR":
                nfr_result = nfr_subclassifier.classify_requirement(requirement)
                nfr_subclass_pred = nfr_result.predicted_label
                nfr_subclass_conf = nfr_result.confidence
                
                print(f"NFR Subclass: {nfr_subclass_pred} (confidence: {nfr_subclass_conf:.2f})")
                
                final_prediction = nfr_subclass_pred  # Use subclass as final prediction for NFRs
                nfr_subclass_counts[nfr_subclass_pred] += 1
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Check correctness with proper logic
            is_correct = False
            
            if true_label == 'FR':
                # True label is Functional - should be classified as FR
                is_correct = (fr_nfr_pred == 'FR')
            elif true_label in ['O', 'PE', 'SE', 'US']:
                # True label is NFR subclass - should be classified as NFR and correct subclass
                is_correct = (fr_nfr_pred == 'NFR' and nfr_subclass_pred == true_label)
            else:
                # Other cases
                is_correct = (final_prediction == true_label)
            
            if is_correct:
                correct_predictions += 1
                print("✅ CORRECT")
            else:
                print("❌ INCORRECT")
                print(f"   Expected: {true_label}")
                print(f"   Got: FR/NFR={fr_nfr_pred}, Subclass={nfr_subclass_pred}")
            
            print(f"Processing Time: {processing_time:.1f}s")
            
            # Store result
            pipeline_results.append({
                'requirement_text': requirement,
                'true_label': true_label,
                'fr_nfr_prediction': fr_nfr_pred,
                'fr_nfr_confidence': fr_nfr_conf,
                'nfr_subclass_prediction': nfr_subclass_pred,
                'nfr_subclass_confidence': nfr_subclass_conf,
                'final_prediction': final_prediction,
                'is_correct': is_correct,
                'processing_time': processing_time
            })
            
            # Rate limiting
            if (i + 1) % 5 == 0 and i + 1 < len(requirements):
                print("⏸️  Pausing for rate limiting...")
                time.sleep(2.0)
        
        # Calculate final metrics
        overall_accuracy = correct_predictions / len(requirements)
        total_time = sum(processing_times)
        avg_time = total_time / len(processing_times)
        
        print("\n" + "=" * 50)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        
        print(f"\n🎯 Overall Performance:")
        print(f"   Total Samples: {len(requirements)}")
        print(f"   Correct Predictions: {correct_predictions}")
        print(f"   Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        
        print(f"\n📋 Classification Distribution:")
        print(f"   Functional Requirements (FR): {fr_count}")
        print(f"   Non-Functional Requirements (NFR): {nfr_count}")
        
        print(f"\n🏷️  NFR Subclass Distribution:")
        for subclass, count in nfr_subclass_counts.items():
            if count > 0:
                full_name = {
                    'O': 'Operational',
                    'PE': 'Performance',
                    'SE': 'Security',
                    'US': 'Usability',
                    'UNKNOWN': 'Unknown/Error'
                }.get(subclass, subclass)
                print(f"   {subclass} ({full_name}): {count}")
        
        print(f"\n⏱️  Performance Statistics:")
        print(f"   Total Processing Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"   Average Time per Requirement: {avg_time:.1f}s")
        print(f"   Estimated Time for 100 samples: {avg_time*100/60:.1f} minutes")
        
        # Detailed error analysis
        print(f"\n🔍 DETAILED ANALYSIS:")
        
        # Analyze by true label
        label_stats = {}
        for result in pipeline_results:
            true_label = result['true_label']
            if true_label not in label_stats:
                label_stats[true_label] = {'total': 0, 'correct': 0}
            
            label_stats[true_label]['total'] += 1
            if result['is_correct']:
                label_stats[true_label]['correct'] += 1
        
        print("   Performance by Class:")
        for label in sorted(label_stats.keys()):
            stats = label_stats[label]
            accuracy = stats['correct'] / stats['total']
            full_name = {
                'FR': 'Functional',
                'O': 'Operational',
                'PE': 'Performance',
                'SE': 'Security',
                'US': 'Usability'
            }.get(label, label)
            print(f"     {label} ({full_name}): {stats['correct']}/{stats['total']} = {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Error analysis
        errors = [r for r in pipeline_results if not r['is_correct']]
        if errors:
            print(f"\n❌ Error Analysis ({len(errors)} errors):")
            
            # FR/NFR level errors
            fr_nfr_errors = []
            subclass_errors = []
            
            for error in errors:
                true_label = error['true_label']
                fr_nfr_pred = error['fr_nfr_prediction']
                
                if true_label == 'FR' and fr_nfr_pred != 'FR':
                    fr_nfr_errors.append(f"FR misclassified as {fr_nfr_pred}")
                elif true_label in ['O', 'PE', 'SE', 'US']:
                    if fr_nfr_pred != 'NFR':
                        fr_nfr_errors.append(f"NFR ({true_label}) misclassified as {fr_nfr_pred}")
                    else:
                        subclass_pred = error['nfr_subclass_prediction']
                        subclass_errors.append(f"{true_label} → {subclass_pred}")
            
            if fr_nfr_errors:
                print(f"   FR/NFR Classification Errors: {len(fr_nfr_errors)}")
                for error in fr_nfr_errors[:3]:  # Show first 3
                    print(f"     - {error}")
            
            if subclass_errors:
                print(f"   NFR Subclassification Errors: {len(subclass_errors)}")
                for error in subclass_errors[:3]:  # Show first 3
                    print(f"     - {error}")
        
        # Save results
        print(f"\n💾 Saving detailed results...")
        save_results(pipeline_results, overall_accuracy, label_stats)
        
        # Performance recommendation
        print(f"\n💡 PERFORMANCE ASSESSMENT:")
        if overall_accuracy >= 0.9:
            print(f"   🎉 Excellent performance! ({overall_accuracy*100:.1f}% accuracy)")
            print("   🚀 Pipeline is ready for production use!")
        elif overall_accuracy >= 0.8:
            print(f"   ✅ Good performance! ({overall_accuracy*100:.1f}% accuracy)")
            print("   💡 Consider minor prompt optimizations for weak classes")
        elif overall_accuracy >= 0.7:
            print(f"   ⚠️  Fair performance. ({overall_accuracy*100:.1f}% accuracy)")
            print("   🔧 Prompt optimization recommended - focus on error classes above")
        else:
            print(f"   🔴 Performance needs significant improvement. ({overall_accuracy*100:.1f}% accuracy)")
            print("   🛠️  Consider revising prompts, examples, or classification strategy")
        
        print(f"\n🎉 PIPELINE EVALUATION COMPLETED!")
        print("📁 Check 'pipeline_evaluation_results/' for detailed results")
        
        return pipeline_results, overall_accuracy, label_stats
        
    except Exception as e:
        print(f"❌ Pipeline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def save_results(pipeline_results, overall_accuracy, label_stats):
    """Save results to CSV and summary files"""
    from datetime import datetime
    
    # Create results directory
    results_dir = Path("pipeline_evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results to CSV
    df = pd.DataFrame(pipeline_results)
    csv_file = results_dir / f"pipeline_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Save summary report
    summary_file = results_dir / f"pipeline_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Complete Hierarchical Pipeline Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(pipeline_results)}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)\n\n")
        
        # Per-class performance
        f.write("Performance by Class:\n")
        for label, stats in sorted(label_stats.items()):
            accuracy = stats['correct'] / stats['total']
            f.write(f"  {label}: {stats['correct']}/{stats['total']} = {accuracy:.3f} ({accuracy*100:.1f}%)\n")
        f.write("\n")
        
        # Distribution stats
        fr_count = len([r for r in pipeline_results if r['fr_nfr_prediction'] == 'FR'])
        nfr_count = len([r for r in pipeline_results if r['fr_nfr_prediction'] == 'NFR'])
        
        f.write("Classification Distribution:\n")
        f.write(f"  FR: {fr_count}\n")
        f.write(f"  NFR: {nfr_count}\n\n")
        
        # NFR subclass distribution
        nfr_subclass_counts = {}
        for result in pipeline_results:
            if result['nfr_subclass_prediction']:
                subclass = result['nfr_subclass_prediction']
                nfr_subclass_counts[subclass] = nfr_subclass_counts.get(subclass, 0) + 1
        
        f.write("NFR Subclass Distribution:\n")
        for subclass, count in nfr_subclass_counts.items():
            f.write(f"  {subclass}: {count}\n")
        
        # Performance stats
        processing_times = [r['processing_time'] for r in pipeline_results]
        f.write(f"\nProcessing Time Statistics:\n")
        f.write(f"  Total: {sum(processing_times):.1f}s\n")
        f.write(f"  Average: {sum(processing_times)/len(processing_times):.1f}s\n")
        f.write(f"  Min: {min(processing_times):.1f}s\n")
        f.write(f"  Max: {max(processing_times):.1f}s\n")
    
    print(f"✅ Results saved to {results_dir}/")

def main():
    """Main function with user interface"""
    print("HIERARCHICAL REQUIREMENTS CLASSIFICATION PIPELINE")
    print("=" * 55)
    
    # Load environment
    load_environment()
    
    # Check API key
    api_key = check_api_key()
    if not api_key:
        return
    
    print("\nSelect evaluation size:")
    print("1. Quick test (10 samples) - ~5 minutes")
    print("2. Medium test (15 samples) - ~8 minutes")
    print("3. Full test (20 samples) - ~10 minutes")
    print("4. Custom size")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            sample_size = 10
            break
        elif choice == "2":
            sample_size = 15
            break
        elif choice == "3":
            sample_size = 20
            break
        elif choice == "4":
            try:
                sample_size = int(input("Enter sample size (max 20): "))
                if sample_size <= 0 or sample_size > 20:
                    print("Sample size must be between 1 and 20")
                    continue
                break
            except ValueError:
                print("Please enter a valid number")
                continue
        else:
            print("Please enter 1-4")
    
    print(f"\n⏰ Starting evaluation with {sample_size} curated test samples...")
    print("This will test: FR/NFR → NFR Subclassification with REAL requirements")
    print("Press Ctrl+C to cancel\n")
    
    try:
        results, accuracy, label_stats = run_pipeline_evaluation(sample_size)
        
        if results is not None:
            print(f"\n🎯 FINAL RESULT: {accuracy*100:.1f}% accuracy on {sample_size} samples")
            print("📈 Per-class breakdown saved in results files")
            
            # Quick recommendation
            if accuracy >= 0.85:
                print("🚀 Excellent! Pipeline shows strong performance!")
            elif accuracy >= 0.75:
                print("💡 Good results! Consider fine-tuning weak classes")
            else:
                print("🔧 Improvement needed - check error analysis for guidance")
                
        else:
            print("❌ Evaluation failed - check errors above")
            
    except KeyboardInterrupt:
        print("\n🛑 Evaluation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()