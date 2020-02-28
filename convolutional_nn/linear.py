import numpy as np
from utils import random_weight_init, one_bias_init

class LinearLayer:
    """
    Arguments -
    1. input_neurons => number of inputs
    2. output_neurons => number of outputs
    """
    def __init__(self, input_neurons, output_neurons):
        self.w = random_weight_init(input_neurons, output_neurons)
        self.b = one_bias_init(output_neurons)
        self.grad_w = np.zeros(self.w.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.grad_w_momentum = np.zeros(self.w.shape)
        self.grad_b_momentum = np.zeros(self.b.shape)

    def forward(self, features):
        """
        Arguments -
          1. features => inputs to linear layer
        """
        self.x = features
        return self.x @ self.w + self.b.T

    def backward(self, dloss):
        """
        Arguments -
          1. dloss => gradient of loss wrt outputs
        """
        self.grad_w = self.x.T @ dloss
        self.grad_b = np.sum(dloss, axis=0, keepdims=True).T
        return self.grad_w, self.grad_b, dloss @ self.w.T

    def zerograd(self):
        self.grad_w = np.zeros_like(self.grad_w)
        self.grad_b = np.zeros_like(self.grad_b)

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Arguments -
          1. learning_rate
          2. momentum_coeff
        """
        self.grad_w_momentum = momentum_coeff * self.grad_w_momentum\
                                + self.grad_w / self.x.shape[0]
        self.grad_b_momentum = momentum_coeff * self.grad_b_momentum\
                + self.grad_b / self.x.shape[0]
        self.w = self.w - learning_rate * self.grad_w_momentum
        self.b = self.b - learning_rate * self.grad_b_momentum

    def get_wb_fc(self):
        """
        Return weights and biases
        """
        return self.w, self.b
