import numpy as np

class CrossEntropy:
    def __init__(self, class_count=None, average=True):
        self._EPS = 1e-8
        self.classes_counts = class_count
        self.average = average
        
    def __call__(self, Y_pred, Y_real):
        '''
        expects: Y_pred - N*D matrix of predictions (N - number of datapoints)
                 Y_real - N*D matrix of one-hot vectors 
        applies softmax before computing negative log likelihood loss
        return a scalar
        '''
        Y_pred = Y_pred - np.max(Y_pred, axis=1)[:, None]
        self.y_pred = Y_pred
        self.y_real = Y_real
        self.N = Y_pred.shape[0]

        # applying softmax function 
        probabilities = np.exp(Y_pred[Y_real.astype(bool)]) / np.sum(np.exp(Y_pred), axis=1)
        logs = np.log(probabilities+self._EPS)

        if self.average:
            return -np.sum(logs) / float(self.N)
        return -np.sum(logs)

    def grad(self):
        '''
        returns gradient with the size equal to the the size of the input vector (self.y_pred)
        '''        
        grad = np.exp(self.y_pred) / np.sum(np.exp(self.y_pred), axis=1)[:, None]
        grad[self.y_real.astype(bool)] -= 1
        grad = grad / self.N
        return grad