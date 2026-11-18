"""A1: mean/max pooled joint-feature global context."""
from model import MLP as Legacy
class MLP(Legacy):
    def __init__(self):super().__init__(architecture='a1_global')
