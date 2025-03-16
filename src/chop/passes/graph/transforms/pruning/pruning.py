import torch

from .random_pruning import random_pruning
from .magnitude_pruning import l1_norm_pruning
from .movement_pruning import movement_pruning
from .hwpq import hwpq_pruning
from .flexround import flexround_pruning

pruning_methods = {
    "random": random_pruning,
    "l1-norm": l1_norm_pruning,
    "movement": movement_pruning,
    "hwpq": hwpq_pruning,
    "flexround": flexround_pruning,
} 