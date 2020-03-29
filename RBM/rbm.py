import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import argparse
import pickle
import os


def binary_data(inp):
    return (inp > 0.5) * 1.


def sigmoid(x):
    """
    Args:
        x: input

    Returns: the sigmoid of x
    """
    return 1 / (1 + np.exp(-x))


def shuffle_corpus(data):
    """shuffle the corpus randomly
    Args:
        data: the image vectors, [num_images, image_dim]
    Returns: The same images with different order
    """
    random_idx = np.random.permutation(len(data))
    return data[random_idx]


class RBM:
    def __init__(self, n_visible, n_hidden, k, lr=0.01, minibatch_size=1):
        """The RBM base class
        Args:
            n_visible: the dimension of visible layer
            n_hidden: the dimension of hidden layer
            k: number of gibbs sampling steps
            lr: learning rate
            minibatch_size: the size of each training batch
            hbias: the bias for the hidden layer
            vbias: the bias for the visible layer
            W: the weights between visible and hidden layer
        """
        # k is the gibbs sampling step
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.hbias = np.zeros(n_hidden)
        self.vbias = np.zeros(n_visible)
        self.W = np.random.normal(0, np.sqrt(
            6.0/(self.n_hidden+self.n_visible)), (n_hidden, n_visible))

    def h_v(self, v):
        """
        Transform the visible vector to hidden vector and compute
        its probability being 1
        Args:
            v: the visible vector
                shape: (dim_v,)
        Returns:
            The probability of output hidden vector, h, being 1, i.e., p(h=1|v)
                shape: (dim_h,)
        """
        return sigmoid(v @ self.W.T + self.hbias)

    def sample_h(self, v):
        """ sample a hidden vector given the distribution p(h=1|v)
        Args:
            v: the visible vector v
                shape: (dim_v,)
        Return:
            The sampling hidden vectors, which are binary in our experiment
            shape: (dim_h,)
        """
        prob_h = self.h_v(v)
        h = np.random.binomial(1, prob_h, (1, self.n_hidden))
        return h, prob_h

    def v_h(self, h):
        """ Transform the hidden vector to visible vector and compute
        its probability being 1
        Args:
            h: the hidden vector
                shape: (dim_h,)
        Returns:
            The probability of output visible vector, v, being 1, i.e., p(v=1|h)
                shape: (dim_v,)
        """
        return sigmoid(h @ self.W + self.vbias)

    def sample_v(self, h):
        """ sample a visible vector given the distribution p(v=1|h)
        Args:
            h: the hidden vector h
                shape: (dim_h,)
        Return:
            The sampling visible vectors, which are binary in our experiment
                shape: (dim_v,)
        """
        prob_v = self.v_h(h)
        v = np.random.binomial(1, prob_v, (1, self.n_visible))
        return v, prob_v

    def gibbs_k(self, v, k=0):
        """ The contrastive divergence k (CD-k) procedure
        Args:
            v: the input visible vector
                shape: (dim_v,)
            k: the number of gibbs sampling steps
                shape: scalar (int)
        Return:
            h0: the hidden vector sample with one iteration
                shape: (dim_h,)
            v0: the input v
                shape: (dim_v,)
            h_sample: the hidden vector sample with k iterations
                shape: (dim_h,)
            v_sample: the visible vector sample with k iterations
                shape: (dim_v,)
            prob_h: the prob of hidden being 1 after k iterations
                shape: (dim_h,)
            prob_v: the prob of visible being 1 after k itersions
                shape: (dim_v,)
        """
        v0 = binary_data(v)
        h0, _ = self.sample_h(v0)
        h_sample = h0.copy()
        v_sample = v0.copy()
        prob_h = self.h_v(v_sample)
        prob_v = self.v_h(h_sample)
        for i in range(k if k > 0 else self.k):
            v_sample, prob_v = self.sample_v(h_sample)
            h_sample, prob_h = self.sample_h(v_sample)
        return h0, v0, h_sample, v_sample, prob_h, prob_v

    def update(self, x):
        """ update our RBM with input x
        Args:
            x: the input data x
                shape: (1,dim_v,) (1, dim_v)
        """
        h0, v0, h_sample, v_sample, prob_h, prob_v = self.gibbs_k(x)

        hvx = self.h_v(x)
        hvvs = self.h_v(v_sample)

        self.hbias = self.hbias + self.lr * (hvx - hvvs)
        self.hbias = self.hbias.reshape(-1)
        self.vbias = self.vbias + self.lr * (x - v_sample)
        self.vbias = self.vbias.reshape(-1)

        pos = hvx.reshape(-1, 1) @ x.reshape(1, -1)
        neg = hvvs.reshape(-1, 1) @ v_sample.reshape(1, -1)
        self.W = self.W + self.lr * (pos - neg)

    def eval(self, X):
        """ Compute reconstruction error
        Args:
            X: the input X
                shape: [num_X, dim_X]
        Return:
            The reconstruction error
                shape: a scalar
        """
        re = 0
        for x in X:
            _, _, _, v_sample, _, _ = self.gibbs_k(x, k=1)
            re += np.sqrt(np.sum((x - v_sample)**2))
        return re / X.shape[0]


if __name__ == "__main__":

    np.seterr(all='raise')

    parser = argparse.ArgumentParser(description='data, parameters, etc.')
    parser.add_argument(
        '-train', type=str, help='training file path',
        default='data/digitstrain.txt')
    parser.add_argument(
        '-valid', type=str, help='validation file path',
        default='data/digitsvalid.txt')
    parser.add_argument('-test', type=str, help="test file path",
                        default="data/digitstest.txt")
    parser.add_argument('-max_epoch', type=int,
                        help="maximum epoch", default=100)
    parser.add_argument('-n_hidden', type=int,
                        help="num of hidden units", default=250)
    parser.add_argument('-k', type=int, help="CD-k sampling", default=3)
    parser.add_argument('-lr', type=float, help="learning rate", default=0.01)
    parser.add_argument('-minibatch_size', type=int,
                        help="minibatch_size", default=1)

    args = parser.parse_args()

    train_data = np.genfromtxt(args.train, delimiter=",")
    train_X = train_data[:, :-1]
    train_X = binary_data(train_X)
    train_Y = train_data[:, -1]

    valid_data = np.genfromtxt(args.valid, delimiter=",")
    valid_X = valid_data[:, :-1]
    valid_X = binary_data(valid_X)
    valid_Y = valid_data[:, -1]

    test_data = np.genfromtxt(args.test, delimiter=",")
    test_X = test_data[:, :-1]
    test_X = binary_data(test_X)
    test_Y = test_data[:, -1]

    n_visible = train_X.shape[1]

    print("input dimension is " + str(n_visible))

    old_cross_entropy = float('inf')
    old_classification_error = float('inf')

    plot_epoch_ce_train = []
    plot_epoch_ce_valid = []
    plot_epoch_re_train = []
    plot_epoch_re_valid = []

    n_visible = train_X.shape[1]
    rbm = RBM(n_visible=n_visible, n_hidden=args.n_hidden, k=args.k, lr=args.lr)

    for epoch in range(args.max_epoch):
        shuffled_data = shuffle_corpus(train_X)
        for x in shuffled_data:
            rbm.update(x)
        te = rbm.eval(shuffled_data)
        ve = rbm.eval(valid_X)
        plot_epoch_re_train.append(te)
        plot_epoch_re_valid.append(ve)
        print(f"Epoch {epoch+1} :: Train Error {te} :: Valid Error {ve}")
