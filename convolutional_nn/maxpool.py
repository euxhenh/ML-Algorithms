import numpy as np

class MaxPool:
    """
    Arguments -
    1. filter_shape => (filter_height, filter_width)
    2. stride
    """
    def __init__(self, filter_shape, stride):
        self.fh, self.fw = filter_shape
        self.st = stride

    def forward(self, inputs):
        """
        Arguments -
          1. inputs => inputs to maxpool forward are outputs from conv layer
        """
        B, C, H, W = inputs.shape
        self.grad = np.zeros(inputs.shape)
        out = np.zeros((B, C, 1 + (H - self.fh) // self.st,
                            1 + (W - self.fw) // self.st))
        for b in range(B):
            for c in range(C):
                for j, jj in enumerate(range(0, H-self.fh+self.st, self.st)):
                    for i, ii in enumerate(range(0, W-self.fw+self.st, self.st)):
                        block = inputs[b, c, jj:jj+self.fh, ii:ii+self.fw]
                        p, q = divmod(np.argmax(block.reshape(-1)), self.fw)
                        self.grad[b, c, jj+p, ii+q] = 1
                        out[b, c, j, i] = block[p, q]
        return out

    def backward(self, dloss):
        """
        Arguments -
          1. dloss => derivative loss wrt output
        """
        B, C, H, W = self.grad.shape
        for b in range(B):
            for c in range(C):
                for j, jj in enumerate(range(0, H-self.fh+self.st, self.st)):
                    for i, ii in enumerate(range(0, W-self.fw+self.st, self.st)):
                        block = self.grad[b, c, jj:jj+self.fh, ii:ii+self.fw]
                        self.grad[b, c, jj:jj+self.fh, ii:ii+self.fw] *= dloss[b, c, j, i]
        return self.grad
