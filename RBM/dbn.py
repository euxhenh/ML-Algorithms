import math
import numpy as np
import numpy as np
import pickle
from rbm import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid


class DBN:
    def __init__(self, n_v, layers, k=1):
        """ the DBN class
        Args:
            n_v: the visible layer dimension
            layers: a list, the dimension of each hidden layer, e.g,, [500, 784]
            k: the number of gibbs sampling steps for each RBM
        """
        self.n_v = n_v
        self.layers = layers
        self.k = k

    def train(self, train_data, valid_data, epochs=1, lr=0.01):
        """ The training process of a DBN, basically we train RBMs one by one
        Args:
            train_data: the train images, numpy matrix
            valid_data: the valid images, numpy matrix
            epochs: the trainig epochs for each RBM
            lr: learning rate
        """

        # Construct RMBs
        self.rbms = []
        r = RBM(n_visible=self.n_v,
                n_hidden=self.layers[0],
                k=self.k,
                lr=lr)
        self.rbms.append(r)
        for i in range(1, len(self.layers)):
            r = RBM(n_visible=self.layers[i-1],
                    n_hidden=self.layers[i],
                    k=self.k,
                    lr=lr)
            self.rbms.append(r)

        # zero lists for reconstruction errors
        self.te_list = np.zeros((len(self.rbms), epochs))
        self.ve_list = np.zeros((len(self.rbms), epochs))

        # iterate over all rbms
        for i in range(len(self.rbms)):
            if i > 0:  # get new data
                train = []
                valid = []
                for x in train_data:
                    h, _ = self.rbms[i-1].sample_h(x)
                    train.append(h)
                for x in valid_data:
                    h, _ = self.rbms[i-1].sample_h(x)
                    valid.append(h)
                train = np.array(train)
                valid = np.array(valid)
            else:
                train = train_data
                valid = valid_data

            # iterate over all epochs
            for epoch in range(epochs):
                shuff = shuffle_corpus(train)
                for x in shuff:
                    self.rbms[i].update(x)

                te = self.rbms[i].eval(train)
                ve = self.rbms[i].eval(valid)
                self.te_list[i][epoch] = te
                self.ve_list[i][epoch] = ve
                print(f"Epoch {epoch+1} :: RMB {i+1} :: " +
                      f"Train Error {te} :: Valid Error {ve}")


class MNIST_DBN:
    def __init__(self, n_v, layers, gibbs_step):
        self.dbn = DBN(n_v, layers, gibbs_step)

        train_data = np.genfromtxt('../data/digitstrain.txt', delimiter=",")
        self.train_X = train_data[:, :-1]
        self.train_X = self.train_X[:900]
        self.train_Y = train_data[:, -1]

        valid_data = np.genfromtxt('../data/digitsvalid.txt', delimiter=",")
        self.valid_X = valid_data[:, :-1][:300]
        self.valid_Y = valid_data[:, -1]

        test_data = np.genfromtxt('../data/digitstest.txt', delimiter=",")
        self.test_X = test_data[:, :-1][:300]
        self.test_Y = test_data[:, -1]

        self.train_X = binary_data(self.train_X)
        self.valid_X = binary_data(self.valid_X)
        self.test_X = binary_data(self.test_X)

    def train(self, n_epoch=1, learning=0.01):
        self.dbn.train(self.train_X, self.valid_X, n_epoch, learning)


mnist_dbn = None
if __name__ == "__main__":

    np.seterr(all='raise')
    plt.close('all')

    v = 28 * 28
    layers = [500, 28 * 28]
    lr = 0.01
    epochs = 50
    k = 100

    mnist_dbn = MNIST_DBN(v, layers, k)
    mnist_dbn.train(n_epoch=epochs, learning=lr)

    def plot_images(images, cols=3, cmap='gray'):
        rows = (len(images) + cols - 1) // cols
        fig, ax = plt.subplots(rows, cols)
        for i, image in enumerate(images):
            ax[i//cols][i%cols].imshow(image, cmap=cmap)
            ax[i//cols][i%cols].get_xaxis().set_ticks([])
            ax[i//cols][i%cols].get_yaxis().set_ticks([])
        for i in range(len(images), rows*cols):
            ax[i//cols][i%cols].get_xaxis().set_ticks([])
            ax[i//cols][i%cols].get_yaxis().set_ticks([])
            ax[i//cols][i%cols].axis('off')
        fig.set_size_inches(cols*10, rows*10)
        plt.show()

    ims = []
    for i in range(100):
        np.random.seed()
        v2 = np.random.binomial(1, .1, (1, 500))
        _, _, hs, vs, hp, vp = mnist_dbn.dbn.rbms[1].gibbs_k(v2)
        v1, _ = mnist_dbn.dbn.rbms[0].sample_v(vs)
        ims.append(v1.reshape((28, 28)))
    ims = np.array(ims)

    plot_images(ims, 10)
