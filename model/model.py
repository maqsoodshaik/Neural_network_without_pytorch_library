import activations
import layers
import numpy as np

class Model:
    def __init__(self, components) -> None:
        '''
        expects a list of components of the model in order with which they must be applied
        '''
        self.components = components
        self.state = 0

    def forward(self, x):
        '''
        performs forward pass on the input x using all components from self.components
        '''
        for component in self.components:
            x = component(x)
        return x
        
    def backward(self, in_grad):
        '''
        expects in_grad - a gradient of the loss w.r.t. output of the model
        in_grad must be of the same size as the output of the model

        returns dictionary, where 
            key - index of the component in the component list
            value - value of the gradient for that component
        '''
        num_components = len(self.components)
        grads = {}
        for i in range(num_components-1, -1, -1):
            component = self.components[i]
            if component.get_type() == 'activation':
                in_grad = component.grad(in_grad)
            elif component.get_type() == 'layer':
                weights_grad, in_grad = component.grad(in_grad)
                grads[i] = weights_grad
            else:
                raise Exception
        return grads

    def update_parameters(self, grads, lr):
        '''
        performs one gradient step with learning rate lr for all components
        ''' 
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.components[i].weights = self.components[i].weights - lr * grad

    def sgd_momentum(self, grads, lr, momentum):
        # your implementation of SGD with momentum goes here
        if self.state == 0:
            self.grad_past = dict.fromkeys(grads, 0)
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.grad_past[i] = (momentum*self.grad_past[i])+grad
            self.components[i].weights = self.components[i].weights - lr * self.grad_past[i] 
        self.state=1
    
    def ada_grad(self, grads, lr, eps=1e-8):
        # your implementation of AdaGrad goes here
        if self.state == 0:
            self.sum_grad = dict.fromkeys(grads, 0)
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.sum_grad[i] += grad**2.0
            alpha = np.full(grad.shape, lr)
            alpha = alpha/(eps+np.sqrt(self.sum_grad[i]))
            self.components[i].weights = self.components[i].weights - (alpha * grad)
        self.state=1
    
    def adam(self, grads, lr, eps=1e-8,beta_1 =0.9 , beta_2=0.999):
        # your implementation of Adam goes here
        if self.state == 0:
            self.grad_past = dict.fromkeys(grads, 0)
            self.sum_grad = dict.fromkeys(grads, 0)
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.grad_past[i] = (beta_1*self.grad_past[i])+((1-beta_1)*grad)
            self.sum_grad[i] = (beta_2*self.sum_grad[i])+((1-beta_2)*(grad**2.0))
            grad_past_corrected = self.grad_past[i]/(1-(beta_1)**(self.state+1))
            sum_grad_corrected = self.sum_grad[i]/(1-(beta_2)**(self.state+1))
            self.components[i].weights = self.components[i].weights - ((lr * grad_past_corrected)/(np.sqrt(sum_grad_corrected)+eps))
        self.state+=1