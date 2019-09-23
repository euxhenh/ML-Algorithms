"""
By Euxhen Hasanaj
ERM implementation for Decision Stumps.

Runs in O(d*m*log(m)) where d is the number of features and m is the number
of data points.

Finds the feature j and threshold theta which best model the sample data.
In other words, finds the best half-ray for the best feature.
"""

def erm_ds(X, Y, D=[]):
    """
    X - training data
    Y - labels for the training data
    D - probability vector (uniform distribution if not set)

    Returns objective values for j, theta, b which minimize
        val = \sum_{i, y_i=1}{ D_i * I(x_{i,j} > theta) } +
                        \sum_{i, y_i=-1}{ D_i * I(x_{i,j} <= theta) }
    or minimize (1 - val).

    If (val) is minimized, then return b = 1, otherwise,
                                    if (1 - val) is minimized return b = -1
    """
    assert(len(X) > 0)
    assert(len(X) == len(Y))
    assert(len(D) == len(X) or len(D) == 0)

    objective_val = float('inf')
    objective_theta = 0
    objective_index = -1
    objective_b = 1

    if len(D) == 0:
        D = [1/len(X)] * len(X)   # Uniform distribution over the sample

    d = len(X[0])       # Number of features per data point
    for j in range(d):  # Consider the j-th feature
        val, theta, b = best_ray_for_feature(j, X, Y, D)
        if val < objective_val:
            objective_val = val
            objective_theta = theta
            objective_index = j
            objective_b = b

    return objective_index, objective_theta, objective_b

def best_ray_for_feature(j, X, Y, D=[]):
    """
    Sort the x_i based on given feature j and set theta_i to be halfway
    between x_i and x_{i+1}. Calculate loss for each such theta_i and
    notice that after each step, the loss changes only by a single x_i.
    This can reduce the performing time of the algorithm to m*log(m) 
    required for sorting only.
    """
    assert(len(X) > 0)
    assert(len(X) == len(Y))
    assert(len(D) == len(X) or len(D) == 0)

    objective_val = float('inf')
    objective_theta = 0
    objective_b = 1

    d = len(X[0])       # Number of features per data point
    m = len(X)          # Number of data points

    if len(D) == 0:
        D = [1/m] * m   # Uniform distribution over the sample

    # Sort according to the j-th feature
    X, Y, D = zip(*sorted(zip(X, Y, D), key = lambda row: row[0][j]))

    # Boundaries
    theta_l = X[0][j] - 1/2
    theta_u = X[m-1][j] + 1/2

    # Sum of D[i] over all i for which Y[i] = 1
    val = sum([D[i] if Y[i] == 1 else 0 for i in range(m)])

    # The case when theta is smaller than all x_i
    if val < objective_val or (1 - val) < objective_val:
        objective_val = min(val, 1 - val)
        objective_theta = theta_l
        objective_b = 1 if val < 1 - val else -1

    # The case when theta is within the range of x_i
    for i in range(m - 1):
        val -= Y[i] * D[i]
        if val < objective_val or (1 - val) < objective_val:
            if X[i][j] < X[i+1][j]:  # Only update if value of x changed
                objective_val = min(val, 1 - val)
                objective_theta = 1/2 * (X[i][j] + X[i+1][j])
                objective_b = 1 if val < 1 - val else -1

    val -= Y[m - 1] * D[m - 1]
    # The case when theta is greater than all x_i
    if val < objective_val or (1 - val) < objective_val:
        objective_val = min(val, 1 - val)
        objective_theta = theta_u
        objective_b = 1 if val < 1 - val else -1

    return objective_val, objective_theta, objective_b

def read_data(name="data"):
    fl = open(name, "r")
    lines = fl.readlines()
    X = []
    Y = []

    for line in lines:
        x = line.strip().split(' ')
        x = [float(i) for i in x]
        y = int(x[-1])
        del x[-1]
        X.append(x)
        Y.append(y)
    fl.close()
    return X, Y

if __name__ == '__main__':
    X, Y = read_data()
    print(erm_ds(X, Y))
