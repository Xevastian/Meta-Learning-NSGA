"""nsga2 package

Keep this module minimal to avoid importing submodules (and heavy
dependencies) at package import time. Import submodules explicitly when
needed, e.g. `from nsga2.meta_learner import MetaLearner`.

Available Algorithms:
  - NSGA-II: Non-dominated Sorting Genetic Algorithm II
  - MOEA/D: Multiobjective Evolutionary Algorithm Based on Decomposition
"""

from .MetaNSGA2 import MetaLearningNSGA2
from .MetaMOEAD import MetaLearningMOEAD

__all__ = [
    'nsga2',
    'moead',
    'nondominated_sort',
    'MetaLearningNSGA2',
]
