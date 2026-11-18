"""A3 with only the full-person summary width increased to 128.

The preceding hidden layer stays 128: 4420 -> 128 -> 128.
The scoring input expands accordingly; all other widths stay unchanged.
"""
from model_a3 import MLP as A3

class MLP(A3):
    def __init__(self):
        super().__init__(context_dim=128)
