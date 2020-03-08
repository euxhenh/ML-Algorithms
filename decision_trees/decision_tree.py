import random
import time
import queue
from matplotlib import pyplot as plt
from math import log2

age = ["10-19", "20-29", "30-39", "40-49",
       "50-59", "60-69", "70-79", "80-89", "90-99"]
menopause = ["lt40", "ge40", "premeno"]
tumor_size = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
              "40-44", "45-49", "50-54", "55-59"]
inv_nodes = ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26",
             "27-29", "30-32", "33-35", "36-39"]
node_caps = ["no", "yes"]
deg_malig = ["1", "2", "3"]
breast = ["left", "right"]
breast_quad = ["left_up", "left_low", "right_up", "right_low", "central"]
irradiat = ["no", "yes"]

features = [age, menopause, tumor_size, inv_nodes, node_caps, deg_malig,
            breast, breast_quad, irradiat]

labels = ["no-recurrence-events", "recurrence-events"]
NO_REC = 0
REC = 1


def preprocess_data(filename):
    # Treat the data as ordinal
    X, Y = [], []
    with open(filename, "r") as fl:
        lines = fl.readlines()
        for line in lines:
            row = line.strip().split(',')
            y = labels.index(row[0])
            x = [features[i-1].index(row[i]) for i in range(1, len(row))]
            X.append(x)
            Y.append(y)
    return X, Y


def train_error(a):
    return min(a, 1 - a)


def gini_index(a):
    return 2 * a * (1 - a)


def information_gain(a):
    return 0 if a < 1e-10 or a > 1-1e-10 else -a * log2(a) - (1 - a) * log2(1 - a)


def gain(X, Y, i, gain_measure):
    m = len(X)
    assert(m > 0)
    # List of counters for each value of feature i
    # I.e., [{ X : X=vi }] for all i
    xvi = [0] * len(features[i])
    # List of counters for positive labels for each value of feature i
    # I.e., [{ Y : Y=REC and X=vi }] for all i
    yvi = [0] * len(features[i])
    for x, y in zip(X, Y):
        xvi[x[i]] += 1
        yvi[x[i]] += 1 if y == REC else 0

    PyREC = sum(yvi) / m
    HY = gain_measure(PyREC)
    HX = 0
    for xi, yi in zip(xvi, yvi):
        if xi > 0:
            # Entropy of conditional probabilities, P(Y = REC | x[i] = vi)
            HX += xi/m * gain_measure(yi / xi)

    return HY - HX


class DecisionNode:
    # Decision tree node
    def __init__(self, feature):
        # Some feature j
        # If node is a leaf, then this will be REC or NO_REC
        self.feature = feature
        # Possible values vi of feature j
        self.branches = {}

    def add_node(self, vi, node):
        self.branches[vi] = node

    def isleaf(self):
        return True if len(self.branches) == 0 else False

    def infer(self, x):
        if self.isleaf():
            return self.feature
        # Find the right branch and continue from there
        return self.branches[x[self.feature]].infer(x)


def kchoices(X, Y, A, k, gain_measure):
    """
    Returns the feature which maximizes gain. The feature is selected from
    a subset of k elements of A which is chosen uniformly at random.
    gain_measure is a function pointer
    """
    Ak = A.copy() if len(A) <= k else random.sample(A, k)
    maxGain, argmax = float('-inf'), None
    for i in Ak:
        m = gain(X, Y, i, gain_measure)
        if m > maxGain:
            maxGain, argmax = m, i
    return argmax


def modifiedid3(X, Y, A, k, gain_measure, max_depth=None):
    """
    Works on multivariate features. First calculate a feature j that maximizes
    the gain, and then add a node for all possible values of the feature j.
    The feature j is selected from kchoices, which randomly samples a size k
    subset from A and finds j that maximizes the gain.
    """
    m = len(Y)
    ypos = Y.count(REC)         # Count positively labeled points
    # Reached the end
    if len(A) == 0 or ypos == m or ypos == 0 or max_depth == 0:
        return DecisionNode(REC if ypos >= m/2 else NO_REC)

    j = kchoices(X, Y, A, k, gain_measure)  # The feature that maximizes gain
    root = DecisionNode(j)

    for vi in range(len(features[j])):      # Consider all values vi of j
        Xvi, Yvi = [], []                   # Hold all points with x[j]=vi
        for x, y in zip(X, Y):
            if x[j] == vi:
                Xvi.append(x)
                Yvi.append(y)
        if len(Xvi) == 0:                   # No points with this vi
            root.add_node(vi, DecisionNode(REC if ypos >= m/2 else NO_REC))
        else:                               # Branch out and recur
            Acopy = A.copy()
            Acopy.remove(j)
            root.add_node(vi, modifiedid3(Xvi, Yvi, Acopy, k, gain_measure,
                                          max_depth - 1 if max_depth is not None else None))
    return root

##############################################################################


def getacc(tree, X, Y):
    # Get accuracy
    errors = 0
    for x, y in zip(X, Y):
        errors += (tree.infer(x) != y)
    accuracy = 1 - errors/len(X)
    return accuracy


def differentgains(Xtrain, Ytrain, Xtest, Ytest):
    # Compare different gain measures
    A = list(range(len(Xtrain[0])))
    k = 9

    tree = modifiedid3(Xtrain, Ytrain, A, k, information_gain)
    print("Information gain")
    print('Training accuracy for k =', k, 'is', getacc(tree, Xtrain, Ytrain))
    print('Test accuracy for k =', k, 'is', getacc(tree, Xtest, Ytest))

    tree = modifiedid3(Xtrain, Ytrain, A, k, gini_index)
    print("\nGini index")
    print('Training accuracy for k =', k, 'is', getacc(tree, Xtrain, Ytrain))
    print('Test accuracy for k =', k, 'is', getacc(tree, Xtest, Ytest))

    tree = modifiedid3(Xtrain, Ytrain, A, k, train_error)
    print("\nTrain error")
    print('Training accuracy for k =', k, 'is', getacc(tree, Xtrain, Ytrain))
    print('Test accuracy for k =', k, 'is', getacc(tree, Xtest, Ytest))


def differentk(Xtrain, Ytrain, Xtest, Ytest):
    # Perform many experiments with different k
    A = list(range(len(Xtrain[0])))
    experiments = 1000

    random.seed()
    for k in [1, 2, 9]:
        accuracy_test = 0
        accuracy_train = 0
        now = time.time()
        for experiment in range(experiments):
            tree = modifiedid3(Xtrain, Ytrain, A, k, information_gain)
            accuracy_test += getacc(tree, Xtest, Ytest)
            accuracy_train += getacc(tree, Xtrain, Ytrain)
        end = time.time()
        print(end - now, 'seconds runtime')
        accuracy_test /= experiments
        accuracy_train /= experiments
        print('Test accuracy for k =', k, 'is', getacc(tree, Xtest, Ytest))
        print('Train accuracy for k =', k, 'is', getacc(tree, Xtrain, Ytrain))
        print()


def depth(tree):
    # Find depth of tree
    M = -1
    if len(tree.branches) > 0:
        for node in tree.branches:
            m = depth(tree.branches[node]) + 1
            M = max(M, m)
    return M


def treesize(Xtrain, Ytrain):
    # Find number of nodes being used
    k = 9
    A = list(range(len(Xtrain[0])))
    nodes = 1
    tree = modifiedid3(Xtrain, Ytrain, A, k, train_error)
    q = queue.Queue()
    q.put(tree)
    while not q.empty():
        r = q.get()
        if len(r.branches) > 0:
            for node in r.branches:
                q.put(r.branches[node])
                nodes += 1
    print("Number of nodes is", nodes)
    print("Depth of tree is", depth(tree) + 1)


def differentdepth(Xtrain, Ytrain, Xtest, Ytest):
    k = 9
    A = list(range(len(Xtrain[0])))
    trainacclist = []
    testacclist = []
    experiments = 100
    depths = list(range(9))
    for depth in depths:
        trainacc = 0
        testacc = 0
        for experiment in range(experiments):
            tree = modifiedid3(Xtrain, Ytrain, A, k, train_error, depth)
            trainacc += getacc(tree, Xtrain, Ytrain)
            testacc += getacc(tree, Xtest, Ytest)
        trainacc /= experiments
        testacc /= experiments
        trainacclist.append(trainacc)
        testacclist.append(testacc)
    plt.plot(depths, trainacclist, label='train')
    plt.plot(depths, testacclist, label='test')
    plt.legend(loc='upper left')
    plt.show()


def main():
    Xtrain, Ytrain = preprocess_data("./data/breast-cancer.data.train")
    Xtest, Ytest = preprocess_data("./data/breast-cancer.data.test")

    #differentgains(Xtrain, Ytrain, Xtest, Ytest)
    #differentk(Xtrain, Ytrain, Xtest, Ytest)
    #treesize(Xtrain, Ytrain)
    #differentdepth(Xtrain, Ytrain, Xtest, Ytest)


if __name__ == '__main__':
    main()
