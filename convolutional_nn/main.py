import numpy as np
import pickle as pk

from relu import ReLU
from conv2d import Conv
from maxpool import MaxPool
from linear import LinearLayer
from cross_entropy_loss import SoftMaxCrossEntropyLoss

class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        """
        self.conv1 = Conv(input_shape=(3, 32, 32), filter_shape=(1, 5, 5))
        self.relu1 = ReLU()
        self.pool1 = MaxPool(filter_shape=(2, 2), stride=2)
        self.linear1 = LinearLayer(256, 20)
        self.loss = SoftMaxCrossEntropyLoss()

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        self.batch = inputs.shape[0]
        x = self.conv1.forward(inputs, stride=1, pad=2)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        yhat = self.linear1.forward(x.reshape(x.shape[0], -1))
        l, preds = self.loss.forward(yhat, y_labels, get_predictions=True)
        return l, preds

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the
        forward function
        DO NOT return anything from this function
        """
        d = self.loss.backward()
        dw, db, d = self.linear1.backward(d)
        d = d.reshape(self.batch, 1, 16, 16)
        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        dw, db, d = self.conv1.backward(d)

    def zerograd(self):
        self.linear1.zerograd()
        self.conv1.zerograd()

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the
        computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.linear1.update(learning_rate, momentum_coeff)
        self.conv1.update(learning_rate, momentum_coeff)

def labels2onehot(labels):
    return np.array([[i==lab for i in range(20)]for lab in labels])

if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    We will not test your training and testing loops, however,
    you can generate results here.
    NOTE - Please generate your results using the classes and functions you
    implemented.
    DO NOT implement the network in either Tensorflow or Pytorch to get
    the results.
    Results from these libraries will vary a bit compared to the expected results
    """
    import pickle
    import tqdm

    epochs = 100
    batch_size = 32
    lr = 0.01
    mom = 0.9

    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
        train, test = data["train"], data["test"]
        xtrain, ytrain = train['data'], labels2onehot(train['labels'])
        xtest, ytest = test['data'], labels2onehot(test['labels'])

        conv = ConvNet()

        for epoch in tqdm.tqdm(range(epochs)):
            indices = np.arange(xtrain.shape[0])
            np.random.shuffle(indices)
            bar = tqdm.tqdm(range(0, indices.shape[0], batch_size))
            for i in bar:
                conv.zerograd()
                y = ytrain[indices[i: i + batch_size]]
                l, preds = conv.forward(xtrain[indices[i: i + batch_size], :], y)
                conv.backward()
                conv.update(learning_rate=lr, momentum_coeff=mom)

            train_loss = 0
            train_acc = 0
            bb = 2000
            for i in tqdm.tqdm(range(0, xtrain.shape[0], bb), desc='evaling'):
                tl, preds=conv.forward(xtrain[i:i+bb], ytrain[i:i+bb])
                train_acc+=np.sum(np.argmax(ytrain[i:i+bb, :], axis=-1)==preds)
                train_loss += tl
            train_acc /= xtrain.shape[0]
            train_loss /= xtrain.shape[0]

            test_loss, test_preds = conv.forward(xtest, ytest)
            test_loss /= xtest.shape[0]
            test_acc = np.mean(np.argmax(ytest, axis=-1) == test_preds)

            tqdm.tqdm.write("Epoch {0} :: Train Loss {1:.2f} :: Train Acc {2:.2f} :: Test Loss {3:.2f} :: Test Acc {4:.2f}".format(epoch+1, train_loss, train_acc, test_loss, test_acc))
