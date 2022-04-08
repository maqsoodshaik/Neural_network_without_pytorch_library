import numpy as np

class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, x):
        self.x = x
        return np.multiply(x, (x > 0))

    def get_type(self):
        return 'activation'

    # assign gradient of zero if x = 0 (even though the function is not differentiable at that point)
    def grad(self, in_gradient):
        return np.multiply((self.x > 0), in_gradient)