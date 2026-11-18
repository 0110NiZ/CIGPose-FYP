"""Fixed-order whole-person MLP without uncertainty: 4403 -> 128 -> 64 -> 17."""
from model_direct_with_u import MLP as WholePersonMLP

class MLP(WholePersonMLP):
    def __init__(self):
        super().__init__(include_uncertainty=False)
