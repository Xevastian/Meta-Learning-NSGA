#!/usr/bin/env python
"""Quick test of __init__.py imports"""

try:
    from nsga2 import MetaLearner, Model, nsga2, nondominated_sort
    print("=" * 60)
    print("✓ SUCCESS! All package imports work!")
    print("=" * 60)
    print("\nThe __init__.py fix resolved the import issues!")
    print("\nAvailable imports from nsga2 package:")
    print(f"  • MetaLearner: {type(MetaLearner)}")
    print(f"  • Model: {type(Model)}")
    print(f"  • nsga2 (function): {type(nsga2)}")
    print(f"  • nondominated_sort (function): {type(nondominated_sort)}")
    print("\n✓ Ready to use Meta-Learning NSGA-II!")
    print("=" * 60)
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
