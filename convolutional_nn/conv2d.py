import numpy as np
from utils import (
    random_weight_init_conv,
    im2col,
    im2col_bw,
    one_bias_init
)


class Conv:
    """
    Arguments -
    1. input_shape => (channels, height, width)
    2. filter_shape => (num of filters, filter height, filter width)
    """

    def __init__(self, input_shape, filter_shape):
        self.C, self.H, self.W = input_shape
        self.k_num, self.k_height, self.k_width = filter_shape
        self.w = random_weight_init_conv(self.k_num, self.C,
                                         self.k_height, self.k_width)
        self.b = one_bias_init(self.k_num)
        self.grad_w = np.zeros(self.w.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.grad_w_momentum = np.zeros(self.w.shape)
        self.grad_b_momentum = np.zeros(self.b.shape)

    def forward(self, inputs, stride, pad):
        """
        Arguments -
          1. inputs => input image of dimension (batch_size, channels,
                                                    height, width)
          2. stride => stride of convolution
          3. pad => padding
        """
        self.padding = pad
        self.stride = stride
        self.ishape = inputs.shape
        batch_size, C, H, W = inputs.shape
        self.xcol = im2col(inputs, k_height=self.k_height, k_width=self.k_width,
                           padding=pad, stride=stride)
        self.Hs = 1 + (H + 2 * pad - self.k_height) // stride
        self.Ws = 1 + (W + 2 * pad - self.k_width) // stride

        return (self.w.reshape(self.k_num, -1) @ self.xcol + self.b).reshape(self.k_num, self.Hs, self.Ws, batch_size).transpose(3, 0, 1, 2)

    def backward(self, dloss):
        """
        Arguments -
          1. dloss => derivative of loss wrt output
        """
        a = dloss.transpose(1, 2, 3, 0).reshape(self.k_num, -1, order='A')
        self.grad_w = (a @ self.xcol.T).reshape(self.w.shape)
        self.grad_b = np.sum(dloss, axis=(0, 2, 3)).reshape(self.b.shape)
        dlossw = self.w.reshape(self.k_num, -1).T @ a
        grad_x = im2col_bw(dlossw[:, :], self.ishape,
                           k_height=self.k_height,
                           k_width=self.k_width,
                           padding=self.padding,
                           stride=self.stride)
        return self.grad_w, self.grad_b, grad_x

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
            + self.grad_w / self.ishape[0]
        self.grad_b_momentum = momentum_coeff * self.grad_b_momentum\
            + self.grad_b / self.ishape[0]
        self.w = self.w - learning_rate * self.grad_w_momentum
        self.b = self.b - learning_rate * self.grad_b_momentum

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return self.w, self.b
