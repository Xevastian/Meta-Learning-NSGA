"""
Hybrid Demo: Custom NSGA-II vs PyMOO-based NSGA-II with Meta-Learning

This demonstrates how to use both implementations side-by-side,
allowing you to choose which one to use or compare their performance.
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from nsga2 import nsga2 as custom_nsga2
    from pymoo_nsga2 import pymoo_nsga2, compare_nsga2_implementations
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the nsga2 directory")
    sys.exit(1)


def run_custom_nsga2_demo():
    """Run demo using custom NSGA-II implementation."""
    print("\n" + "="*80)
    print("CUSTOM NSGA-II DEMO (with Meta-Learning)")
    print("="*80)
    print("""
This is your existing custom implementation with:
- Custom non-dominated sorting
- Custom crowding distance calculation
- Meta-learning warm-start
- Adaptive mutation rates based on diversity
- Model-specific crossover and mutation operators
""")
    
    # Example usage
    data_path = '../digit.csv'  # Adjust to your dataset
    
    if os.path.exists(data_path):
        print(f"\nRunning optimization on {data_path}...")
        result = custom_nsga2(
            pop_size=10,
            generations=3,
            data_path=data_path,
            use_warm_start=True,
            adaptive_operators=True,
            seed=42
        )
        print(f"\n✓ Custom NSGA-II completed successfully")
        return result
    else:
        print(f"⚠ Dataset not found at {data_path}")
        return None


def run_pymoo_nsga2_demo():
    """Run demo using PyMOO-based NSGA-II implementation."""
    print("\n" + "="*80)
    print("PYMOO-BASED NSGA-II DEMO (with Meta-Learning)")
    print("="*80)
    print("""
This is the new PyMOO-based implementation with:
- Highly optimized genetic operators (SBX crossover, PM mutation)
- Industry-standard implementation used in benchmark studies
- Meta-learning warm-start integration
- Adaptive mutation rates based on diversity
- Faster convergence in many scenarios
""")
    
    # Example usage
    data_path = '../digit.csv'  # Adjust to your dataset
    
    if os.path.exists(data_path):
        print(f"\nRunning optimization on {data_path}...")
        try:
            result = pymoo_nsga2(
                pop_size=10,
                generations=3,
                data_path=data_path,
                use_warm_start=True,
                adaptive_operators=True,
                seed=42,
                verbose=True
            )
            print(f"\n✓ PyMOO-based NSGA-II completed successfully")
            return result
        except ImportError as e:
            print(f"\n✗ PyMOO not available: {e}")
            print("Install with: pip install pymoo")
            return None
    else:
        print(f"⚠ Dataset not found at {data_path}")
        return None


def run_comparison_demo():
    """Run both implementations and compare results."""
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    print("""
This runs both implementations on the same dataset with the same seed,
allowing you to compare:
- Pareto front quality
- Convergence speed
- Computational efficiency
- Meta-learning benefits
""")
    
    data_path = '../digit.csv'  # Adjust to your dataset
    
    if os.path.exists(data_path):
        print(f"\nRunning comparison on {data_path}...")
        try:
            results = compare_nsga2_implementations(
                data_path=data_path,
                pop_size=10,
                generations=3,
                seed=42,
                use_warm_start=True,
                adaptive_operators=True
            )
            
            print("\n" + "="*80)
            print("COMPARISON RESULTS")
            print("="*80)
            
            custom_pf = results['custom']
            pymoo_pf = results['pymoo']
            
            print(f"\nCustom NSGA-II Runtime: {custom_pf.get('runtime', 'N/A')} seconds")
            print(f"PyMOO NSGA-II Runtime:  {pymoo_pf.get('runtime', 'N/A')} seconds")
            
            print("\n✓ Comparison completed successfully")
            return results
        except Exception as e:
            print(f"\n✗ Comparison failed: {e}")
            return None
    else:
        print(f"⚠ Dataset not found at {data_path}")
        return None


def print_guide():
    """Print usage guide."""
    print("\n" + "="*80)
    print("HYBRID NSGA-II IMPLEMENTATIONS GUIDE")
    print("="*80)
    print("""
You now have TWO implementations of NSGA-II with Meta-Learning:

1. CUSTOM NSGA-II (nsga2.py)
   - Full control over algorithm details
   - Custom genetic operators tailored to your Model class
   - Great for research and experimentation
   - Usage:
       from nsga2 import nsga2
       result = nsga2(pop_size=20, generations=10, data_path='data.csv')

2. PYMOO-BASED NSGA-II (pymoo_nsga2.py)
   - Industry-standard implementation
   - Highly optimized operators (SBX, PM)
   - Better performance benchmarks in many cases
   - Easier to integrate with other PyMOO algorithms
   - Usage:
       from pymoo_nsga2 import pymoo_nsga2
       result = pymoo_nsga2(pop_size=20, generations=10, data_path='data.csv')

3. HYBRID APPROACH
   - Use both implementations
   - Run comparison to benchmark
   - Usage:
       from pymoo_nsga2 import compare_nsga2_implementations
       results = compare_nsga2_implementations(data_path='data.csv')

BOTH IMPLEMENTATIONS SUPPORT:
✓ Meta-learning warm-start (use_warm_start=True)
✓ Adaptive mutation rates (adaptive_operators=True)
✓ Reproducibility (seed parameter)
✓ Dataset similarity matching (dataset_similarity_threshold)
✓ Meta-knowledge persistence (update_meta_db)

CHOOSING BETWEEN IMPLEMENTATIONS:

Use Custom NSGA-II if:
- You need fine-grained control over operators
- You're experimenting with new genetic operator designs
- You need to debug specific evolutionary behavior

Use PyMOO NSGA-II if:
- You want leveraging industry-standard algorithms
- You need faster convergence
- You plan to use other PyMOO algorithms later
- You want benchmark-comparable results

Try Comparison if:
- You want empirical results for your specific dataset
- You're writing a paper (compare both approaches)
- You want to understand the trade-offs
""")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid NSGA-II Demo')
    parser.add_argument('--custom', action='store_true', help='Run custom NSGA-II demo')
    parser.add_argument('--pymoo', action='store_true', help='Run PyMOO NSGA-II demo')
    parser.add_argument('--compare', action='store_true', help='Run comparison of both')
    parser.add_argument('--guide', action='store_true', help='Print usage guide')
    parser.add_argument('--all', action='store_true', help='Run all demos')
    
    args = parser.parse_args()
    
    if not any([args.custom, args.pymoo, args.compare, args.guide, args.all]):
        args.guide = True
    
    if args.guide or args.all:
        print_guide()
    
    if args.custom or args.all:
        run_custom_nsga2_demo()
    
    if args.pymoo or args.all:
        run_pymoo_nsga2_demo()
    
    if args.compare or args.all:
        run_comparison_demo()
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)
