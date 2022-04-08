import numpy as np

class LeakyReLU:
    def __init__(self, alpha= 0.01) -> None:
        self.alpha = alpha

    def __call__(self, x):#
        self.x = x
        return np.maximum(0, x) + self.alpha * np.minimum(0, x)

    def get_type(self):
        return 'activation'
    
    # assign gradient of in_gradient*alpha if x = 0 (even though the function is not differentiable at that point)
    def grad(self, in_gradient):
        return np.multiply((self.x > 0), in_gradient) + np.multiply((self.x <= 0), in_gradient*self.alpha)