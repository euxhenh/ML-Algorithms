"""
By Euxhen Hasanaj

Different implementations of the Gain Measure

X - n times d array
    each row is a feature vector of d dimensions where each feature is binary
    i.e., X[i] \in {0, 1}^d
Y - n times 1 array containing the labels for vectors in X
    i.e., Y[i] \in {-1, 1}
i - index for determining gain at feature i, where i < d
"""
from math import log

def core_gain(X, Y, i, cfunc):
    """
    Cfunc is a function used to extract different information from probabilities
    """
    if len(X) == 0:
        raise "Gain function called with zero length array"
    if len(X) != len(Y):
        raise "Gain function called with different size arrays"
    if i >= len(X[0]):
        raise "Index exceeds array bounds in gain function call"

    x0 = 0              # number of points with (x[i]=0)
    px0 = 0             # probability of the above
    y1x0, y1x1, = 0, 0  # number of points with (y=1 | x[i]=0) and (y=1 | x[i]=1)
    py1x0, py1x1 = 0, 0 # probability of the above
    py1 = 0             # probability that (Y=1)

    # Calculate x0, x1, y1x0, y1x1
    for x, y in zip(X, Y):
        if x[i] == 0:
            x0 += 1
            if y == 1:
                y1x0 += 1
        elif y == 1:
            y1x1 += 1

    # Calculate probabilities
    px0 = x0 / len(X)
    px1 = 1 - px0
    py1 = (y1x0 + y1x1) / len(X)
    if x0 != 0:
        py1x0 = y1x0 / x0
    if x0 != len(X):
        py1x1 = y1x1 / (len(X) - x0)
    
    # Return difference
    return cfunc(py1) - (px0 * cfunc(py1x0) + px1 * cfunc(py1x1))

def cfunc_train_error(a):
    return min(a, 1 - a)

def train_error_gain(X, Y, i):
    return core_gain(X, Y, i, cfunc_train_error)

def cfunc_information_gain(a):
    return 0 if a == 0 or a == 1 else -a * log(a) - (1 - a) * log(1 - a)

def information_gain(X, Y, i):
    return core_gain(X, Y, i, cfunc_information_gain)

def cfunc_gini_index(a):
    return 2 * a * (1 - a)

def gini_index_gain(X, Y, i):
    return core_gain(X, Y, i, cfunc_gini_index)
