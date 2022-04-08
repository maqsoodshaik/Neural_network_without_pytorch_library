import numpy as np

class MSELoss:
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, y_pred):
        # saving the inputs
        self.y_true = y_true
        self.y_pred = y_pred
        self.N = self.y_pred.shape[0]
        return np.sum((y_true - y_pred)**2) / self.N
    
    def grad(self):
        '''
        returns gradient equal to the the size of input vector (self.y_pred)
        '''
        return 1/self.N * 2 * (self.y_true - self.y_pred)