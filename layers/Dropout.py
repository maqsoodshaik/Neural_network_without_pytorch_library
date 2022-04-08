import numpy as np
import copy

class Dropout:
    def __init__(self, layer, p : float = 0.5):
        self.p = p
        self.layer = layer
        self.weights = self.layer.weights
        self.bias = self.layer.bias

    def __call__(self, x):
        '''
            apply inverted dropout
        '''
        out = self.layer(x)
        scaler = 1.0 / (1.0 - self.p)
        self.mask = np.random.binomial(1, 1 - self.p, size=out.shape)
        
        
        return out * scaler * self.mask

    def get_type(self):
        return 'layer'

    def grad(self, in_gradient):
        return self.layer.grad(in_gradient) * self.mask