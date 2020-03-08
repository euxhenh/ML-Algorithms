import numpy as np


class SoftMaxCrossEntropyLoss:
    """
    Softmax and cross entropy loss
    """

    def __init__(self):
        pass

    def forward(self, logits, labels, get_predictions=False):
        """
        Forward pass through softmax and loss function
        Arguments -
          1. logits => pre-softmax scores
          2. labels => true labels of given inputs
          3. get_predictions => If true, the forward function returns
          predictions along with the loss
        """
        self.labels = labels
        self.batch_size = labels.shape[0]
        exps = np.exp(logits)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        if get_predictions:
            preds = np.argmax(logits, axis=-1)
            return -np.sum(labels * np.log(self.probs)), preds
        else:
            return -np.sum(labels * np.log(self.probs))

    def backward(self):
        """
        Return gradient of loss with respect to inputs of softmax
        """
        return (self.probs - self.labels)
