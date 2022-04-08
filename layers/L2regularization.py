import numpy as np

class L2regularization(object):
    """
        Implement the class such that it wraps around a linear layer
        and modifies the backward pass of a regularized linear layer
    """

    def __init__(self, layer, coefficient = 0.01):
        """
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient
        self.layer = layer
        self.weights = self.layer.weights
        self.bias = self.layer.bias
        
    def __call__(self, x):
        """.
        """
        out = x @ self.weights + self.bias
        # save the current input (should be a minibatch)
        self.X = x
        return out
        
    def grad(self, in_gradient):
        """
        """
        m = self.X.shape[1]
        return self.X.T @ in_gradient + (self.coefficient/m) * self.weights, in_gradient @ self.weights.T

    def get_type(self):
        return "layer"