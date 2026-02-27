#!/usr/bin/env python
"""
QUICK START: Meta-Learning NSGA-II
===================================

This script shows the fastest way to get started with Meta-Learning NSGA-II.
"""

import os
import sys

# Add current directory to path so nsga2 package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsga2 import nsga2
from nsga2.meta_learner import MetaLearner


def quick_start():
    """
    Quick start example: Run meta-learning NSGA-II on your data.
    """
    
    # ============ CONFIGURATION ============
    DATASET = 'train.csv'          # Change this to your data file
    POP_SIZE = 20                  # Population size
    GENERATIONS = 10               # Number of generations
    
    # ============ RUN ONCE: Baseline ============
    print("\n" + "="*60)
    print("FIRST RUN: Building Meta-Knowledge Base")
    print("="*60)
    print("\nThis first run optimizes your problem and learns good solutions.")
    print("Subsequent runs will use this knowledge to converge faster.\n")
    
    if not os.path.exists(DATASET):
        print(f"❌ Error: Dataset '{DATASET}' not found!")
        print("\nAvailable CSV files:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  • {f}")
        sys.exit(1)
    
    # First run (no warm start yet, but learns for next run)
    pop1 = nsga2(
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        data_path=DATASET,
        plot_path='optimization_run1.png',
        use_warm_start=False,      # First run has no prior knowledge
        adaptive_operators=True    # But uses adaptive operators
    )
    
    # ============ RUN TWICE: With Meta-Learning ============
    print("\n" + "="*60)
    print("SECOND RUN: Using Meta-Knowledge")
    print("="*60)
    print("\nThis run uses knowledge from the first run.")
    print("You should see faster convergence!\n")
    
    pop2 = nsga2(
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        data_path=DATASET,
        plot_path='optimization_run2.png',
        use_warm_start=True,       # ← Enable warm-start!
        adaptive_operators=True
    )
    
    # ============ PRINT RESULTS ============
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Check the generated plots:
   • optimization_run1.png - First run (learning phase)
   • optimization_run2.png - Second run (using meta-knowledge)
   
2. View meta-knowledge summary:
   • meta_summary.txt - Statistics on learned solutions
   
3. Run again on different problems:
   • Meta-knowledge persists in 'meta_knowledge.pkl'
   • Subsequent runs automatically use it for warm-start
   
4. For detailed analysis:
   • python nsga2/meta_learning_demo.py
   
5. For more information:
   • Read: nsga2/README_META_LEARNING.md
    """)
    
    print("="*60)
    print("✓ Quick start complete!")
    print("="*60 + "\n")


def example_api():
    """
    Example: How to use Meta-Learner directly in your code.
    """
    
    print("\n" + "="*60)
    print("API EXAMPLE: Using MetaLearner Directly")
    print("="*60 + "\n")
    
    from meta_learner import MetaLearner
    
    # Initialize meta-learner
    ml = MetaLearner()
    
    # Get best model type from learned knowledge
    best_model = ml.get_best_model_type()
    print(f"Best model type from history: {best_model}\n")
    
    # Get population diversity metric
    pop = [
        {'accuracy': 0.90, 'size': 100},
        {'accuracy': 0.85, 'size': 200},
        {'accuracy': 0.92, 'size': 150},
    ]
    diversity = ml.compute_population_diversity(pop)
    print(f"Population diversity: {diversity:.3f}")
    print("  0.0 = all identical, 1.0 = very spread out\n")
    
    # Get adaptive mutation rate
    pm = ml.get_adaptive_mutation_rate(diversity)
    print(f"Adaptive mutation rate: {pm:.3f}")
    print(f"  (Base rate is 0.3, adapts based on diversity)\n")
    
    # View meta-knowledge summary
    print("Meta-knowledge statistics:")
    print(f"  • Total solutions learned: {len(ml.meta_knowledge['solutions'])}")
    print(f"  • Model types tracked: {len(ml.meta_knowledge['model_stats'])}")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Meta-Learning NSGA-II Quick Start')
    parser.add_argument('--full', action='store_true', help='Run full quick start')
    parser.add_argument('--api', action='store_true', help='Show API examples')
    
    args = parser.parse_args()
    
    if args.api:
        example_api()
    else:
        # Default: Run quick start
        quick_start()
