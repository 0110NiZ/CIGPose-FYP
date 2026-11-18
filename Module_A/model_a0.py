"""Historical A0: local features plus pose-only global context."""
from model import MLP as Legacy
class MLP(Legacy):
    def __init__(self):super().__init__(architecture='a0')
