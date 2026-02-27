#!/usr/bin/env python
"""
Validation Script: Check Meta-Learning NSGA-II Installation
===========================================================

Run this script to verify that all components are properly installed
and working correctly.
"""

import os
import sys
import traceback

# Add current directory to path so nsga2 package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_files():
    """Verify all required files exist."""
    print("\n" + "="*60)
    print("CHECKING FILES")
    print("="*60 + "\n")
    
    required_files = {
        'nsga2/nsga2.py': 'Core NSGA-II implementation',
        'nsga2/models.py': 'Model definitions',
        'nsga2/trainer.py': 'Model trainer',
        'nsga2/meta_learner.py': 'Meta-learning module',
        'nsga2/meta_learning_demo.py': 'Comparison demo',
        'SUMMARY.md': 'Implementation summary',
        'GETTING_STARTED.md': 'Getting started guide',
        'IMPLEMENTATION_NOTES.md': 'Technical documentation',
        'quick_start.py': 'Quick start script'
    }
    
    all_exist = True
    for filepath, description in required_files.items():
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        print(f"{status} {filepath:<35} ({description})")
        if not exists:
            all_exist = False
    
    return all_exist


def check_imports():
    """Verify all required modules can be imported."""
    print("\n" + "="*60)
    print("CHECKING IMPORTS")
    print("="*60 + "\n")
    
    required_modules = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-Learn',
        'matplotlib': 'Matplotlib'
    }
    
    all_imported = True
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name:<20} ({module})")
        except ImportError as e:
            print(f"✗ {name:<20} ({module}) - {e}")
            all_imported = False
    
    # Check local modules
    print()
    local_modules = [
        ('models', 'nsga2.models'),
        ('nsga2', 'nsga2.nsga2'),
        ('trainer', 'nsga2.trainer'),
        ('meta_learner', 'nsga2.meta_learner')
    ]
    
    for display_name, import_path in local_modules:
        try:
            parts = import_path.split('.')
            module = __import__(import_path)
            for part in parts[1:]:
                module = getattr(module, part)
            print(f"✓ {display_name:<20} (local module)")
        except Exception as e:
            print(f"✗ {display_name:<20} (local module) - {e}")
            all_imported = False
    
    return all_imported


def check_data():
    """Verify dataset exists."""
    print("\n" + "="*60)
    print("CHECKING DATA")
    print("="*60 + "\n")
    
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("✗ No CSV files found!")
        print("\nTo run the demos, you need to provide a CSV dataset.")
        print("Expected columns: features + 'label' column")
        return False
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        size_kb = os.path.getsize(csv_file) / 1024
        print(f"  • {csv_file:<30} ({size_kb:.1f} KB)")
    
    return True


def check_functionality():
    """Test basic functionality."""
    print("\n" + "="*60)
    print("CHECKING FUNCTIONALITY")
    print("="*60 + "\n")
    
    try:
        # Test imports
        from nsga2.meta_learner import MetaLearner
        from nsga2.models import Model
        from nsga2.trainer import Trainer
        from nsga2 import nsga2
        
        print("✓ All modules imported successfully")
        
        # Test MetaLearner instantiation
        ml = MetaLearner(meta_db_path='test_meta.pkl')
        print("✓ MetaLearner instantiated")
        
        # Test Model instantiation
        model = Model()
        print(f"✓ Model created: {model.getModelName()}")
        
        # Test diversity computation
        test_pop = [
            {'accuracy': 0.9, 'size': 100},
            {'accuracy': 0.85, 'size': 200},
        ]
        diversity = ml.compute_population_diversity(test_pop)
        print(f"✓ Diversity computed: {diversity:.3f}")
        
        # Test adaptive mutation rate
        pm = ml.get_adaptive_mutation_rate(diversity)
        print(f"✓ Adaptive Pm computed: {pm:.3f}")
        
        # Clean up test file
        if os.path.exists('test_meta.pkl'):
            os.remove('test_meta.pkl')
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality check failed: {e}")
        traceback.print_exc()
        return False


def check_documentation():
    """Verify documentation quality."""
    print("\n" + "="*60)
    print("CHECKING DOCUMENTATION")
    print("="*60 + "\n")
    
    docs = {
        'SUMMARY.md': 'Implementation summary',
        'GETTING_STARTED.md': 'Getting started guide',
        'IMPLEMENTATION_NOTES.md': 'Technical documentation',
        'nsga2/README_META_LEARNING.md': 'API documentation'
    }
    
    all_present = True
    for filepath, description in docs.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                print(f"✓ {filepath:<40} ({lines} lines, {size} bytes)")
            except Exception as e:
                print(f"⚠ {filepath:<40} (Error reading: {e})")
        else:
            print(f"✗ {filepath:<40} (MISSING)")
            all_present = False
    
    return all_present


def run_validation():
    """Run complete validation."""
    print("\n" + "█"*60)
    print("META-LEARNING NSGA-II - VALIDATION SUITE")
    print("█"*60)
    
    results = {
        'Files': check_files(),
        'Imports': check_imports(),
        'Data': check_data(),
        'Documentation': check_documentation(),
        'Functionality': check_functionality(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60 + "\n")
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou're ready to use Meta-Learning NSGA-II!")
        print("\nQuick start:")
        print("  1. cd nsga2")
        print("  2. python nsga2.py ../train.csv 20 10")
        print("  3. Run again to see speedup!")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above, then run validation again.")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_validation()
    sys.exit(0 if success else 1)
