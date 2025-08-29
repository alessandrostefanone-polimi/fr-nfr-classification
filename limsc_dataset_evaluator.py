"""
Complete Pipeline Evaluator for LIMSC Dataset
=============================================

This script evaluates the complete hierarchical pipeline on the actual
limsc/subclass-classification dataset from HuggingFace.

Usage: python limsc_dataset_evaluator.py [--samples N] [--inspect-only]
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Tuple, Dict, Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def load_environment():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

def inspect_limsc_dataset():
    """Inspect the limsc dataset structure and return mapping info"""
    try:
        from datasets import load_dataset
        
        print("🔍 INSPECTING LIMSC DATASET STRUCTURE")
        print("=" * 45)
        
        dataset = load_dataset("limsc/subclass-classification")
        print(f"✅ Dataset loaded: {list(dataset.keys())} splits")
        
        # Use first available split
        split_name = list(dataset.keys())[0]
        split_data = dataset[split_name]
        df = split_data.to_pandas()
        
        print(f"📊 Split '{split_name}': {df.shape[0]} samples, {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\n📝 Sample data:")
        print(df.head())
        
        # Analyze columns
        for col in df.columns:
            unique_vals = df[col].unique()
            print(f"\n📊 Column '{col}':")
            print(f"   Type: {df[col].dtype}")
            print(f"   Unique values: {len(unique_vals)}")
            
            if len(unique_vals) <= 20:
                print(f"   Values: {sorted(unique_vals)}")
                
                # Show distribution
                dist = df[col].value_counts()
                print(f"   Distribution: {dict(dist)}")
            else:
                print(f"   Sample values: {sorted(unique_vals)[:10]}...")
                
                # For text columns, show length stats
                if df[col].dtype == 'object':
                    lengths = df[col].astype(str).str.len()
                    print(f"   Length stats: min={lengths.min()}, max={lengths.max()}, avg={lengths.mean():.1f}")
        
        return dataset, df
        
    except Exception as e:
        print(f"❌ Error inspecting dataset: {e}")
        return None, None

def create_dataset_mapping(df: pd.DataFrame) -> Tuple[str, str, Dict]:
    """Create mapping based on dataset structure analysis"""
    
    print(f"\n🎯 CREATING DATASET MAPPING")
    print("-" * 30)
    
    # Identify text column (longest average text)
    text_candidates = []
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_length = df[col].astype(str).str.len().mean()
            text_candidates.append((col, avg_length))
    
    # Sort by average length, longest first
    text_candidates.sort(key=lambda x: x[1], reverse=True)
    text_column = text_candidates[0][0] if text_candidates else df.columns[0]
    
    # Identify label column (smallest number of unique values)
    label_candidates = []
    for col in df.columns:
        if col != text_column:
            unique_count = len(df[col].unique())
            label_candidates.append((col, unique_count))
    
    # Sort by unique count, smallest first
    label_candidates.sort(key=lambda x: x[1])
    label_column = label_candidates[0][0] if label_candidates else df.columns[-1]
    
    print(f"🎯 Selected text column: '{text_column}'")
    print(f"🏷️ Selected label column: '{label_column}'")
    
    # Create label mapping
    unique_labels = sorted(df[label_column].unique())
    print(f"📊 Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Create mapping based on label type
    if all(isinstance(label, (int, float)) for label in unique_labels):
        print("📋 Numeric labels detected - creating mapping:")
        
        # Common mappings for requirements classification
        label_mapping_options = [
            # Option 1: Standard NFR classification
            {0: 'FR', 1: 'O', 2: 'PE', 3: 'SE', 4: 'US'},
            # Option 2: Alternative mapping
            {1: 'FR', 2: 'O', 3: 'PE', 4: 'SE', 5: 'US'},
            # Option 3: Binary then subclass
            {0: 'FR', 1: 'NFR'},  # If only 2 classes
            # Option 4: Direct subclass mapping
            {0: 'O', 1: 'PE', 2: 'SE', 3: 'US'}
        ]
        
        # Choose best mapping based on number of unique labels
        if len(unique_labels) == 2:
            label_mapping = {unique_labels[0]: 'FR', unique_labels[1]: 'NFR'}
        elif len(unique_labels) == 4:
            # Map to NFR subclasses
            nfr_classes = ['O', 'PE', 'SE', 'US']
            label_mapping = {unique_labels[i]: nfr_classes[i] for i in range(len(unique_labels))}
        elif len(unique_labels) == 5:
            # Map to FR + NFR subclasses
            all_classes = ['FR', 'O', 'PE', 'SE', 'US']
            label_mapping = {unique_labels[i]: all_classes[i] for i in range(len(unique_labels))}
        else:
            # Default cycling mapping
            all_classes = ['FR', 'O', 'PE', 'SE', 'US']
            label_mapping = {}
            for i, label in enumerate(unique_labels):
                label_mapping[label] = all_classes[i % len(all_classes)]
        
        print("💡 Created label mapping:")
        for orig, mapped in label_mapping.items():
            count = (df[label_column] == orig).sum()
            print(f"   {orig} → {mapped} ({count} samples)")
    
    else:
        # String labels - try to normalize
        label_mapping = {}
        for label in unique_labels:
            label_str = str(label).upper()
            if label_str in ['FR', 'FUNCTIONAL']:
                label_mapping[label] = 'FR'
            elif label_str in ['NFR', 'NON-FUNCTIONAL']:
                label_mapping[label] = 'NFR'
            elif label_str in ['O', 'OPERATIONAL']:
                label_mapping[label] = 'O'
            elif label_str in ['PE', 'PERFORMANCE']:
                label_mapping[label] = 'PE'
            elif label_str in ['SE', 'SECURITY']:
                label_mapping[label] = 'SE'
            elif label_str in ['US', 'USABILITY']:
                label_mapping[label] = 'US'
            else:
                label_mapping[label] = label_str
        
        print("💡 Created string label mapping:")
        for orig, mapped in label_mapping.items():
            count = (df[label_column] == orig).sum()
            print(f"   '{orig}' → '{mapped}' ({count} samples)")
    
    return text_column, label_column, label_mapping

def load_limsc_dataset_with_mapping(sample_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Load limsc dataset with intelligent mapping"""
    
    print("📂 LOADING LIMSC DATASET WITH MAPPING")
    print("=" * 40)
    
    try:
        # Inspect dataset first
        dataset, df = inspect_limsc_dataset()
        if dataset is None or df is None:
            raise Exception("Could not load dataset")
        
        # Create mapping
        text_column, label_column, label_mapping = create_dataset_mapping(df)
        
        # Extract data
        requirements = df[text_column].tolist()
        raw_labels = df[label_column].tolist()
        
        # Apply mapping
        mapped_labels = []
        for label in raw_labels:
            if label in label_mapping:
                mapped_labels.append(label_mapping[label])
            else:
                mapped_labels.append(str(label))  # Fallback
        
        # Clean data - remove invalid entries
        clean_requirements = []
        clean_labels = []
        
        for req, label in zip(requirements, mapped_labels):
            req_str = str(req).strip()
            # Skip if requirement is too short, empty, or looks like an ID
            if (len(req_str) > 15 and 
                req_str.lower() not in ['nan', 'none', ''] and
                not req_str.startswith('promise_')):
                clean_requirements.append(req_str)
                clean_labels.append(label)
        
        print(f"✅ Cleaned dataset: {len(clean_requirements)} valid requirements")
        
        # Show final distribution
        label_counts = {}
        for label in clean_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"📊 Final label distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(clean_labels)) * 100
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        # Sample if requested
        if sample_size and sample_size < len(clean_requirements):
            # Stratified sampling to maintain label distribution
            sampled_req = []
            sampled_labels = []
            
            # Calculate samples per class
            samples_per_class = max(1, sample_size // len(label_counts))
            
            for label in label_counts.keys():
                # Get indices for this label
                label_indices = [i for i, l in enumerate(clean_labels) if l == label]
                
                # Sample from this label
                n_samples = min(samples_per_class, len(label_indices))
                if n_samples > 0:
                    import random
                    random.seed(42)  # Reproducible sampling
                    sampled_indices = random.sample(label_indices, n_samples)
                    
                    for idx in sampled_indices:
                        sampled_req.append(clean_requirements[idx])
                        sampled_labels.append(clean_labels[idx])
            
            # Fill remaining slots randomly if needed
            while len(sampled_req) < sample_size and len(sampled_req) < len(clean_requirements):
                import random
                idx = random.randint(0, len(clean_requirements) - 1)
                if clean_requirements[idx] not in sampled_req:
                    sampled_req.append(clean_requirements[idx])
                    sampled_labels.append(clean_labels[idx])
            
            clean_requirements = sampled_req[:sample_size]
            clean_labels = sampled_labels[:sample_size]
            
            print(f"📊 Sampled to {len(clean_requirements)} requirements")
        
        return clean_requirements, clean_labels
        
    except Exception as e:
        print(f"❌ Error loading limsc dataset: {e}")
        print("🔄 Falling back to sample data")
        
        # Fallback to sample data
        from nfr_subclass_classifier import get_sample_nfr_data
        return get_sample_nfr_data()

def run_limsc_evaluation(sample_size: Optional[int] = None):
    """Run evaluation on limsc dataset"""
    
    print("🚀 LIMSC DATASET EVALUATION")
    print("=" * 35)
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return None
    
    try:
        # Import components
        print("📦 Importing pipeline components...")
        from langchain_fr_nfr_pipeline import RequirementClassifier
        from nfr_subclass_classifier import LLMNFRSubclassifier
        
        # Initialize pipeline
        print("🔧 Initializing pipeline...")
        fr_nfr_classifier = RequirementClassifier(
            model_type="deepseek",
            model_name="deepseek-chat",
            use_few_shot=True
        )
        
        nfr_subclassifier = LLMNFRSubclassifier(
            model_type="deepseek",
            model_name="deepseek-chat",
            api_key=api_key,
            use_few_shot=True,
            temperature=0.0
        )
        
        # Load dataset
        requirements, true_labels = load_limsc_dataset_with_mapping(sample_size)
        
        print(f"📊 Starting evaluation on {len(requirements)} requirements...")
        print(f"⏱️ Estimated time: {len(requirements) * 0.3:.1f} minutes")
        
        # Run evaluation
        results = []
        correct = 0
        
        for i, (requirement, true_label) in enumerate(zip(requirements, true_labels)):
            start_time = time.time()
            
            if (i + 1) % 10 == 0:
                print(f"📊 Progress: {i+1}/{len(requirements)} ({(i+1)/len(requirements)*100:.1f}%)")
            
            # Step 1: FR/NFR classification
            fr_nfr_result = fr_nfr_classifier.classify_requirement(requirement)
            fr_nfr_pred = fr_nfr_result.predicted_label
            
            final_pred = fr_nfr_pred
            nfr_subclass_pred = None
            
            # Step 2: NFR subclassification if needed
            if fr_nfr_pred == "NFR":
                nfr_result = nfr_subclassifier.classify_requirement(requirement)
                nfr_subclass_pred = nfr_result.predicted_label
                final_pred = nfr_subclass_pred  # Use subclass as final for NFRs
            
            # Check correctness
            is_correct = False
            if true_label == "0":
                is_correct = (fr_nfr_pred == 'FR')
            elif true_label == "1":
                is_correct = (final_pred == 'O')
            elif true_label == "2":
                is_correct = (final_pred == 'PE')
            elif true_label == "3":
                is_correct = (final_pred == 'SE')
            elif true_label == "4":
                is_correct = (final_pred == 'US')

            if is_correct:
                correct += 1
            
            processing_time = time.time() - start_time
            
            results.append({
                'requirement': requirement,
                'true_label': true_label,
                'fr_nfr_pred': fr_nfr_pred,
                'nfr_subclass_pred': nfr_subclass_pred,
                'final_pred': final_pred,
                'correct': is_correct,
                'time': processing_time
            })
            
            # Rate limiting
            if (i + 1) % 5 == 0:
                time.sleep(1.5)
        
        # Calculate metrics
        accuracy = correct / len(results)
        avg_time = np.mean([r['time'] for r in results])
        
        print(f"\n📊 LIMSC DATASET EVALUATION RESULTS")
        print("=" * 40)
        print(f"🎯 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"⏱️ Average Processing Time: {avg_time:.1f}s per requirement")
        print(f"📊 Total Requirements: {len(results)}")
        print(f"✅ Correct: {correct}")
        print(f"❌ Incorrect: {len(results) - correct}")
        
        # Per-class analysis
        class_stats = {}
        for result in results:
            true_label = result['true_label']
            if true_label not in class_stats:
                class_stats[true_label] = {'total': 0, 'correct': 0}
            class_stats[true_label]['total'] += 1
            if result['correct']:
                class_stats[true_label]['correct'] += 1
        
        print(f"\n📋 Per-Class Performance:")
        for label in sorted(class_stats.keys()):
            stats = class_stats[label]
            class_acc = stats['correct'] / stats['total']
            print(f"   {label}: {stats['correct']}/{stats['total']} = {class_acc:.3f} ({class_acc*100:.1f}%)")
        
        # Save results
        save_limsc_results(results, accuracy, class_stats)
        
        return results, accuracy, class_stats
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def save_limsc_results(results, accuracy, class_stats):
    """Save LIMSC evaluation results"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("limsc_evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    df = pd.DataFrame(results)
    csv_file = results_dir / f"limsc_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Save summary
    summary_file = results_dir / f"limsc_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("LIMSC Dataset Evaluation Summary\n")
        f.write("=" * 35 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Total Requirements: {len(results)}\n")
        f.write(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n\n")
        
        f.write("Per-Class Performance:\n")
        for label, stats in sorted(class_stats.items()):
            class_acc = stats['correct'] / stats['total']
            f.write(f"  {label}: {stats['correct']}/{stats['total']} = {class_acc:.3f}\n")
    
    print(f"💾 Results saved to {results_dir}/")

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Evaluate pipeline on LIMSC dataset")
    parser.add_argument('--samples', type=int, help='Number of samples to evaluate')
    parser.add_argument('--inspect-only', action='store_true', help='Only inspect dataset structure')
    
    args = parser.parse_args()
    
    # Load environment
    load_environment()
    
    if args.inspect_only:
        inspect_limsc_dataset()
        return
    
    print("LIMSC DATASET PIPELINE EVALUATION")
    print("=" * 40)
    
    sample_size = args.samples
    if not sample_size:
        print("Select evaluation size:")
        print("1. Small test (25 samples) - ~12 minutes")
        print("2. Medium test (50 samples) - ~25 minutes")
        print("3. Large test (100 samples) - ~50 minutes")
        print("4. Full dataset (all samples) - variable time")
        print("5. Custom size")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            sample_size = 25
        elif choice == "2":
            sample_size = 50
        elif choice == "3":
            sample_size = 100
        elif choice == "4":
            sample_size = None  # Full dataset
        elif choice == "5":
            try:
                sample_size = int(input("Enter sample size: "))
            except ValueError:
                print("Invalid input, using 25 samples")
                sample_size = 25
        else:
            print("Invalid choice, using 25 samples")
            sample_size = 25
    
    print(f"\n🚀 Starting LIMSC evaluation with {sample_size or 'all'} samples...")
    
    try:
        results, accuracy, class_stats = run_limsc_evaluation(sample_size)
        
        if results:
            print(f"\n🎉 EVALUATION COMPLETED!")
            print(f"📈 Final Accuracy: {accuracy*100:.1f}% on LIMSC dataset")
            print(f"📁 Detailed results saved in limsc_evaluation_results/")
        else:
            print("❌ Evaluation failed")
            
    except KeyboardInterrupt:
        print("\n🛑 Evaluation cancelled by user")

if __name__ == "__main__":
    main()