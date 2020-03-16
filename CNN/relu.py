import numpy as np


class ReLU:
    """
    ReLU non-linearity
    """

    def __init__(self):
        pass

    def forward(self, x, train=True):
        # IMPORTANT the autograder assumes that you call
        # np.random.uniform(0,1,x.shape) exactly once in this function
        self.x = x
        return np.where(x < 0, 0, x)

    def backward(self, dLoss_dout):
        return np.where(self.x > 0, 1, 0) * dLoss_dout
