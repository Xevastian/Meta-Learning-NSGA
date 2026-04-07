"""
Multi-Algorithm Comparison: NSGA-II vs MOEA/D vs SMS-EMOA vs Random Search

This script provides a unified interface to run and compare all four baselines:
1. NSGA-II: Dominance-based ranking + crowding distance (with meta-learning)
2. MOEA/D: Decomposition-based with weighted subproblems (pure baseline)
3. SMS-EMOA: Hypervolume-based selection (pure baseline)
4. Random Search: Uniform random sampling (simple baseline)
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

try:
    from nsga2 import nsga2
    from moead import moead
    from sms_emoa import sms_emoa
    from random_search import random_search
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def format_size(size_bytes):
    """Convert bytes to human-readable format (KB or MB)"""
    if size_bytes == float('inf') or size_bytes < 0:
        return "N/A"
    if size_bytes >= 1024**2:  # MB
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:  # KB
        return f"{size_bytes / 1024:.2f} KB"
    else:  # Bytes
        return f"{size_bytes:.0f} B"


# ============================================================================
# Comparison Framework
# ============================================================================

class AlgorithmComparison:
    """Framework for comparing MOO algorithms."""
    
    def __init__(self, data_path: str, pop_size: int = 20, generations: int = 10, 
                 seed: int = 42, verbose: bool = True):
        """
        Initialize comparison.
        
        Args:
            data_path: Path to dataset
            pop_size: Population size
            generations: Number of generations
            seed: Random seed
            verbose: Print progress
        """
        self.data_path = data_path
        self.pop_size = pop_size
        self.generations = generations
        self.seed = seed
        self.verbose = verbose
        self.results = {}
        
    def run_nsga2(self) -> Dict:
        """Run NSGA-II with meta-learning."""
        if self.verbose:
            print("\n" + "="*80)
            print("RUNNING: NSGA-II (Dominance-Ranking + Crowding Distance)")
            print("="*80)
            print("Algorithm: Generational | Selection: Tournament")
            print("Ranking: Non-dominated sort | Diversity: Crowding distance\n")
        
        start = time.time()
        result = nsga2(
            pop_size=self.pop_size,
            generations=self.generations,
            data_path=self.data_path,
            use_warm_start=True,
            adaptive_operators=True,
            seed=self.seed,
            verbose=self.verbose
        )
        runtime = time.time() - start
        
        return {
            'name': 'NSGA-II',
            'algorithm': 'Generational',
            'selection': 'Tournament + Ranking',
            'result': result,
            'runtime': runtime,
            'pareto_size': len([ind for ind in result if hasattr(ind.get('model'), 'getModelName')])
        }
    
    def run_moead(self) -> Dict:
        """Run MOEA/D baseline (no meta-learning)."""
        if self.verbose:
            print("\n" + "="*80)
            print("RUNNING: MOEA/D (Decomposition-based Baseline)")
            print("="*80)
            print("Algorithm: Generational | Selection: Neighborhood-based")
            print("Decomposition: Weighted scalar subproblems | Aggregation: Tchebycheff")
            print("Status: PURE BASELINE (No Meta-Learning)\n")
        
        start = time.time()
        result = moead(
            pop_size=self.pop_size,
            generations=self.generations,
            data_path=self.data_path,
            seed=self.seed,
            verbose=self.verbose
        )
        runtime = time.time() - start
        
        return {
            'name': 'MOEA/D',
            'algorithm': 'Generational',
            'selection': 'Neighborhood-based',
            'result': result,
            'runtime': runtime,
            'pareto_size': len([ind for ind in result if hasattr(ind.get('model'), 'getModelName')])
        }
    
    def run_sms_emoa(self) -> Dict:
        """Run SMS-EMOA baseline (no meta-learning)."""
        if self.verbose:
            print("\n" + "="*80)
            print("RUNNING: SMS-EMOA (S-Metric Selection Baseline)")
            print("="*80)
            print("Algorithm: Steady-state | Selection: Hypervolume-based")
            print("Diversity: S-metric (Hypervolume) | Offspring: One per generation")
            print("Status: PURE BASELINE (No Meta-Learning)\n")
        
        start = time.time()
        result = sms_emoa(
            pop_size=self.pop_size,
            generations=self.generations,
            data_path=self.data_path,
            seed=self.seed,
            verbose=self.verbose
        )
        runtime = time.time() - start
        
        return {
            'name': 'SMS-EMOA',
            'algorithm': 'Steady-state',
            'selection': 'Hypervolume-based',
            'result': result,
            'runtime': runtime,
            'pareto_size': len([ind for ind in result if hasattr(ind.get('model'), 'getModelName')])
        }
    
    def run_random_search(self) -> Dict:
        """Run Random Search baseline."""
        if self.verbose:
            print("\n" + "="*80)
            print("RUNNING: Random Search (Simple Baseline)")
            print("="*80)
            print("Algorithm: Uniform random sampling")
            print("Selection: None (all samples retained)")
            print("Status: SIMPLE BASELINE (No Search Strategy)\n")
        
        # For fair comparison, use pop_size * generations samples
        n_samples = self.pop_size * self.generations * 2
        
        start = time.time()
        result = random_search(
            n_samples=n_samples,
            data_path=self.data_path,
            seed=self.seed,
            verbose=self.verbose
        )
        runtime = time.time() - start
        
        return {
            'name': 'Random Search',
            'algorithm': 'Sampling-based',
            'selection': 'None (Random)',
            'result': result,
            'runtime': runtime,
            'pareto_size': len([ind for ind in result if hasattr(ind.get('model'), 'getModelName')])
        }
    
    def run_all(self) -> Dict:
        """Run all algorithms."""
        self.results = {
            'nsga2': self.run_nsga2(),
            'moead': self.run_moead(),
            'sms_emoa': self.run_sms_emoa(),
            'random_search': self.run_random_search()
        }
        return self.results
    
    def print_comparison_table(self):
        """Print comparison table of all algorithms."""
        if not self.results:
            print("No results to compare. Run algorithms first.")
            return
        
        print("\n" + "="*100)
        print("ALGORITHM COMPARISON SUMMARY")
        print("="*100)
        
        algorithms_info = [
            {
                'name': 'NSGA-II',
                'type': 'Generational',
                'selection': 'Tournament',
                'diversity': 'Crowding Distance',
                'comp_per_gen': '2*pop'
            },
            {
                'name': 'MOEA/D',
                'type': 'Generational',
                'selection': 'Neighborhood',
                'diversity': 'Decomposition',
                'comp_per_gen': '2*pop'
            },
            {
                'name': 'SMS-EMOA',
                'type': 'Steady-state',
                'selection': 'Hypervolume',
                'diversity': 'S-metric',
                'comp_per_gen': '1'
            },
            {
                'name': 'Random Search',
                'type': 'Sampling-based',
                'selection': 'None',
                'diversity': 'None',
                'comp_per_gen': 'All'
            }
        ]
        
        print(f"\n{'Algorithm':<20} {'Type':<15} {'Selection':<15} {'Diversity':<20} {'Offspring/Gen':<15} {'Runtime':<12}")
        print("-" * 110)
        
        for info in algorithms_info:
            alg_key = info['name'].replace(' ', '_').replace('-', '').lower()
            if alg_key in self.results or info['name'].lower().replace(' ', '_') in self.results:
                # Try multiple key formats
                result = None
                for key in [alg_key, info['name'].lower().replace(' ', '_'), info['name'].lower().replace('-', '_')]:
                    if key in self.results:
                        result = self.results[key]
                        break
                
                if result:
                    runtime = result['runtime']
                    print(f"{info['name']:<20} {info['type']:<15} {info['selection']:<15} "
                          f"{info['diversity']:<20} {info['comp_per_gen']:<15} {runtime:>10.2f}s")
        
        print("\n" + "="*110)
        print("ALGORITHM CHARACTERISTICS")
        print("="*110)
        
        print("""
NSGA-II (WITH Meta-Learning):
  ✓ Pros: Simple, well-understood, WITH warm-start & adaptive operators
  ✓ Cons: O(N²) complexity in dominance checks
  • Features: Meta-knowledge database, warm-start, adaptive mutation
  • Best for: General-purpose with meta-learning benefits
  • Complexity: O(N² × generations)

MOEA/D (Pure Baseline):
  ✓ Pros: Scalable to many objectives, neighborhood promotes diversity
  ✓ Cons: Requires weight vector generation, sensitive to decomposition
  • Status: Pure baseline algorithm, NO meta-learning
  • Best for: Many-objective problems, diversity-focused search
  • Complexity: O(N × K × generations) where K = neighborhood size

SMS-EMOA (Pure Baseline):
  ✓ Pros: Steady-state evolution, hypervolume-based selection
  ✓ Cons: Hypervolume computation is expensive
  • Status: Pure baseline algorithm, NO meta-learning
  • Best for: Expensive fitness evaluations, quality-focused optimization
  • Complexity: Higher per-generation but fewer total evaluations

RANDOM SEARCH (Simple Baseline):
  ✓ Pros: Simple, no algorithm overhead, good baseline reference
  ✓ Cons: No guided search, poor solutions, inefficient
  • Status: Simple baseline for lower-bound comparison
  • Best for: Establishing performance reference
  • Complexity: O(n_samples) evaluations, no search structure
        """)
    
    def extract_pareto_front(self, result: List[Dict]) -> List[Tuple[float, float]]:
        """Extract (accuracy, size) pairs from result."""
        # Extract non-dominated solutions
        points = []
        for ind in result:
            if isinstance(ind, dict) and 'accuracy' in ind and 'size' in ind:
                points.append((ind['accuracy'], ind['size']))
        
        # Sort by accuracy descending
        points = sorted(set(points), key=lambda x: (-x[0], x[1]))
        
        # Filter dominated points
        pareto = []
        for p in points:
            dominated = False
            for q in pareto:
                if q[0] >= p[0] and q[1] <= p[1]:
                    if q[0] > p[0] or q[1] < p[1]:
                        dominated = True
                        break
            if not dominated:
                pareto = [q for q in pareto if not (p[0] >= q[0] and p[1] <= q[1])]
                pareto.append(p)
        
        return sorted(pareto, key=lambda x: (-x[0], x[1]))
    
    def print_detailed_results(self):
        """Print detailed results for each algorithm."""
        if not self.results:
            print("No results to display.")
            return
        
        for alg_key in ['nsga2', 'moead', 'sms_emoa', 'random_search']:
            if alg_key not in self.results:
                continue
            
            alg_result = self.results[alg_key]
            name = alg_result['name']
            
            print(f"\n{'='*80}")
            print(f"DETAILED RESULTS: {name}")
            print(f"{'='*80}")
            print(f"Runtime: {alg_result['runtime']:.2f} seconds")
            print(f"Algorithm Type: {alg_result['algorithm']}")
            print(f"Selection Type: {alg_result['selection']}")


def print_guide():
    """Print usage guide."""
    print("""
    
╔════════════════════════════════════════════════════════════════════════════════╗
╔════════════════════════════════════════════════════════════════════════════════╗
║  MULTI-OBJECTIVE OPTIMIZATION: FOUR ALGORITHM COMPARISON                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

AVAILABLE ALGORITHMS:

1️⃣  NSGA-II (Non-dominated Sorting Genetic Algorithm II, WITH Meta-Learning)
   📊 Approach: Rank-based dominance + Crowding Distance
   🔄 Evolution: Generational | 🧠 Features: Warm-start, Adaptive operators
   ⏱️  Complexity: O(N² × generations)
   ✅ Use when: You need evolutionary algorithm with meta-learning benefits
   
   from nsga2 import nsga2
   nsga2(pop_size=20, generations=10, data_path='data.csv', use_warm_start=True)

2️⃣  MOEA/D (Decomposition-based Multi-Objective Evolutionary Algorithm)
   📊 Approach: Decomposition into weighted subproblems
   🔄 Evolution: Generational | 🧠 Features: NONE (Pure baseline)
   ⏱️  Complexity: O(N × K × generations)
   ✅ Use when: You want baseline decomposition-based algorithm
   
   from moead import moead
   moead(pop_size=20, generations=10, data_path='data.csv')

3️⃣  SMS-EMOA (S-Metric Selection NSGA-II)
   📊 Approach: Hypervolume-based selection
   🔄 Evolution: Steady-state | 🧠 Features: NONE (Pure baseline)
   ⏱️  Complexity: Higher per-generation but fewer evals
   ✅ Use when: You have expensive fitness evaluations
   
   from sms_emoa import sms_emoa
   sms_emoa(pop_size=20, generations=10, data_path='data.csv')

4️⃣  RANDOM SEARCH (Simple Baseline)
   📊 Approach: Uniform random sampling
   🔄 Evolution: None (Sampling) | 🧠 Features: NONE (Simple baseline)
   ⏱️  Complexity: O(n_samples) evaluations
   ✅ Use when: You need lower-bound performance reference
   
   from random_search import random_search
   random_search(n_samples=200, data_path='data.csv')

META-LEARNING FEATURES (NSGA-II Only):
✓ Warm-start from meta-knowledge database (use_warm_start=True)
✓ Adaptive mutation rates based on population diversity
✓ Population diversity tracking
✓ Pareto front persistence to meta-knowledge for future runs

QUICK COMPARISON:
═══════════════════════════════════════════════════════════════════════════════════
             NSGA-II          MOEA/D          SMS-EMOA         Random Search
───────────────────────────────────────────────────────────────────────────────────
Scalability  Moderate         High            Low              Very Low
Solution Q   Good             Good            Excellent        Poor
Search       Guided (GA)      Guided (Decomp) Guided (HV)       None
Speed        Fast             Medium          Slow             Very Fast
Meta-Learn   ✅ Yes            ❌ No            ❌ No             ❌ No
───────────────────────────────────────────────────────────────────────────────────

RUNNING THE COMPARISON:
python multi_algorithm_comparison.py <data_path> --compare

PYTHON USAGE:
from multi_algorithm_comparison import AlgorithmComparison

comp = AlgorithmComparison(data_path='data.csv', pop_size=20, generations=10)
comp.run_all()
comp.print_comparison_table()
comp.print_detailed_results()
    """)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Algorithm MOO Comparison')
    parser.add_argument('data_path', nargs='?', help='Path to dataset')
    parser.add_argument('--compare', action='store_true', help='Run all algorithms')
    parser.add_argument('--nsga2', action='store_true', help='Run only NSGA-II')
    parser.add_argument('--moead', action='store_true', help='Run only MOEA/D')
    parser.add_argument('--sms-emoa', action='store_true', help='Run only SMS-EMOA')
    parser.add_argument('--random-search', action='store_true', help='Run only Random Search')
    parser.add_argument('--guide', action='store_true', help='Print guide')
    parser.add_argument('--pop-size', type=int, default=20, help='Population size')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if not args.data_path and not args.guide:
        args.guide = True
    
    if args.guide:
        print_guide()
    
    if args.data_path:
        if not os.path.exists(args.data_path):
            print(f"Error: Dataset not found at {args.data_path}")
            sys.exit(1)
        
        comparison = AlgorithmComparison(
            data_path=args.data_path,
            pop_size=args.pop_size,
            generations=args.generations,
            seed=args.seed,
            verbose=True
        )
        
        if args.compare or (not args.nsga2 and not args.moead and not args.sms_emoa):
            comparison.run_all()
            comparison.print_comparison_table()
            comparison.print_detailed_results()
        else:
            if args.nsga2:
                comparison.results['nsga2'] = comparison.run_nsga2()
            if args.moead:
                comparison.results['moead'] = comparison.run_moead()
            if args.sms_emoa:
                comparison.results['sms_emoa'] = comparison.run_sms_emoa()
            if args.random_search:
                comparison.results['random_search'] = comparison.run_random_search()
            
            if comparison.results:
                comparison.print_comparison_table()
                comparison.print_detailed_results()

