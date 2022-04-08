import numpy as np

class Softmax:
    def __init__(self):
        pass
    
    def __call__(self, x):
        self.x = x  
        self.out = np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
        return self.out

    def get_type(self):
        return 'activation'