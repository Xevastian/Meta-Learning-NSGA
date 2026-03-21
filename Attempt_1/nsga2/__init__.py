"""nsga2 package

Keep this module minimal to avoid importing submodules (and heavy
dependencies) at package import time. Import submodules explicitly when
needed, e.g. `from nsga2.meta_learner import MetaLearner`.
"""

from .MetaNSGA2 import MetaLearningNSGA2

__all__ = [
    'nsga2',
    'nondominated_sort',
    'MetaLearningNSGA2',
]
