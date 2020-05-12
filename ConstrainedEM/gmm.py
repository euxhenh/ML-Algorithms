import pickle

import mnist
import seaborn as sns
import numpy as np
from numpy.testing import assert_array_almost_equal as aae
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.stats import multivariate_normal as mn

import sklearn
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
#import umap

from unionfind import UnionFind
from utils import scatter_points


def init_centers(X, n_clusters):
    """
    Run k-means++ to initialize centroids.
    Since we will be comparing to k-means, it's fair for both methods to have
    the same initialization method.

    Taken from scikit-learn.
    """
    random_state = check_random_state(None)
    n_samples = X.shape[0]

    x_squared_norms = row_norms(X, squared=True)

    centers = _k_init(
        X, n_clusters,
        random_state=random_state,
        x_squared_norms=x_squared_norms)

    return centers


class GMM:
    def __init__(self, n_clusters, n_steps, eps=1e-20):
        self.n_clusters = n_clusters
        self.n_steps = n_steps
        self.eps = eps

    def _initialize(self):
        """
        Initializes self.alpha, self.mu, self.sigma, self.w
        """
        self.alpha = np.ones((self.n_clusters)) / self.n_clusters
        self.mu = self.X[np.random.choice(np.arange(self.n), self.n_clusters)]
        self.sigma = np.ones((self.n_clusters, self.d))
        self.chunklet_w = np.zeros((self.n_chunklets, self.n_clusters))

        #centers = init_centers(X, self.n_clusters)
        #dists = cdist(X, centers)
        #labels = np.argmin(dists, axis=1)

        #unq_labels, self.alpha = np.unique(labels, return_counts=True)

        #self.alpha = np.zeros(self.n_clusters)
        #self.mu = np.zeros((self.n_clusters, d))
        # Using diagonal variance
        #self.sigma = np.zeros((self.n_clusters, d))

        # for i, lbl in enumerate(unq_labels):
        #    cur_pts = np.where(labels == lbl)
        #    self.alpha[i] = cur_pts[0].shape[0]
        #    # initialize means
        #    self.mu[i, :] = np.mean(X[cur_pts], axis=0)

        #    centered = (X[cur_pts] - self.mu[i])**2
        #    centered = np.sum(centered, axis=0) / centered.shape[0]
        #    # initialize vars
        #    self.sigma[i, :] = self.alpha[i] * centered

        #self.alpha /= n

        # self._validate_sigma()

        #self.chunklet_w = np.zeros((self.chunklets.shape[0], self.n_clusters))

    def _transitive_closure(self):
        self.uf = UnionFind(np.arange(self.n))
        for link in self.ml:
            self.uf.union(link[0], link[1])
        self.chunklets = np.array(
            [np.array(list(i)) for i in self.uf.components()])
        self.n_chunklets = self.chunklets.shape[0]
        print(f'Found a total of {self.n_chunklets} chunklets out of '
                f'{self.n} points.')
        self.chunklet_shapes = np.array([i.shape[0] for i in self.chunklets])
        self.chunklet_shapes = self.chunklet_shapes.reshape(-1, 1)
        self.chunklet_means = np.array(
            [np.mean(self.X[i], axis=0) for i in self.chunklets])
        assert self.chunklet_means.shape == (self.n_chunklets, self.d)

    def fit(self, X, ml):
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X.copy()
        self.ml = ml.copy()

        self._transitive_closure()
        self._initialize()

        self.scores = []
        self.lls = []

        for step in range(self.n_steps):
            self.e_step()
            self.m_step()
            self.scores.append(self.score())
            self.lls.append(self.ll/self.n)
            print(f"Step {step+1} :: LL {self.lls[-1]} :: Score {self.scores[-1]}")
            if len(self.lls) >= 2 and np.abs(self.lls[-1] - self.lls[-2]) < 1e-2:
                print("Converged")
                break

    def get_labels(self):
        chunk_labels = np.argmax(self.chunklet_w, axis=1).astype(np.int)
        labels = np.zeros(self.n)
        for i, chunk in enumerate(self.chunklets):
            labels[chunk] = chunk_labels[i]
        return labels.astype(np.int)

    def llhood(self):
        ll = 0
        for i, chunklet in enumerate(self.chunklets):
            for j in range(self.n_clusters):
                numerator = mn.pdf(
                    self.X[chunklet], self.mu[j], np.diag(self.sigma[j]))
                ll += np.sum(np.log(numerator + self.eps), axis=0) *\
                        self.chunklet_w[i,j]
                ll += np.log(self.alpha[j] + self.eps) * self.chunklet_w[i,j]
        return ll

    def e_step(self):
        self.ll = 0

        for i, chunklet in enumerate(self.chunklets):
            denominator = 0
            numerators = []
            for j in range(self.n_clusters):
                try:
                    numerator = mn.pdf(
                        self.X[chunklet], self.mu[j], np.diag(self.sigma[j]))
                except:
                    print("Found non singular matrix.")
                    print(f"Smallest value {np.min(self.sigma[j])}")
                    self.sigma[j] += self.eps
                    numerator = mn.pdf(
                        self.X[chunklet], self.mu[j], np.diag(self.sigma[j]))

                self.ll += np.sum(np.log(numerator + self.eps), axis=0) *\
                        self.chunklet_w[i,j]
                self.ll += np.log(self.alpha[j] + self.eps) *\
                        self.chunklet_w[i,j]

                numerator = np.prod(numerator, axis=0)
                numerator *= self.alpha[j]
                denominator += numerator
                self.chunklet_w[i, j] = numerator
            self.chunklet_w[i, :] /= (denominator + self.eps)
            #assert np.abs(self.chunklet_w[i, :].sum() - 1) < eps,\
            #    np.abs(self.chunklet_w[i, :].sum())

    def m_step(self):
        self.alpha = self.chunklet_w.sum(axis=0) / self.n_chunklets

        for j in range(self.n_clusters):
            den = 0
            temp_mu = np.zeros((1, self.d))

            numfrac = self.chunklet_w[:, j, np.newaxis] * self.chunklet_shapes
            den = np.sum(numfrac, axis=0, keepdims=True)
            temp_mu = np.sum(self.chunklet_means * numfrac, axis=0)
            self.mu[j] = temp_mu / den

            diff_sq = (self.X - self.mu[j])**2
            temp_sigma = np.zeros((1, self.d))
            for i in range(self.n_chunklets):
                # calc sigmanew
                signew = diff_sq[self.chunklets[i]]
                signew = np.sum(signew, axis=0, keepdims=True)
                signew /= self.chunklet_shapes[i]
                temp_sigma += signew * numfrac[i]

            self.sigma[j] = temp_sigma / den

    def score(self):
        labels = self.get_labels()
        if len(np.unique(labels)) > 1:
            return silhouette_score(self.X, labels)
        else:
            return 1


if __name__ == '__main__':
    #x = mnist.train_images()
    #y = mnist.train_labels()
    from load_cifar10 import load_cifar10
    X, y = load_cifar10()

    X = X.reshape(X.shape[0], -1)[:200] / 255
    pca = PCA(n_components=50)
    X = pca.fit_transform(X)
    # with open("dumps/pca_emb_200.pkl", "rb") as f:
    #    X = pickle.load(f)
    #X_emb = umap.UMAP(n_components=2).fit_transform(X)
    # with open("dumps/2d_emb.pkl", "rb") as f:
    #    X_emb = pickle.load(f)

    gmm = GMM(n_clusters=10, n_steps=5)
    ml = np.array([(1, 2), (2, 3), (3, 4), (4, 5), (7, 9), (50, 51)])
    gmm.fit(X, ml)

    #scatter_points(X_emb[:, 0], X_emb[:, 1], gmm.labels)
