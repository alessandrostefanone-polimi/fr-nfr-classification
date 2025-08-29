#!/usr/bin/env python3
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
