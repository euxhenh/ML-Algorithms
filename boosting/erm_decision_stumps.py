def erm_ds(X, Y, D=[]):
    """
    X - training data
    Y - labels for the training data
    D - probability vector
    """
    objective_val = float('inf')
    objective_theta = 0
    objective_index = -1
    objective_b = 1

    d = len(X[0])       # Number of features per data point
    m = len(X)          # Number of data points

    if len(D) == 0:
        D = [1/m] * m

    for j in range(d):                  # Consider the j-th feature
        X, Y = zip(*sorted(zip(X, Y), key = lambda row: row[0][j]))

        theta_l = X[0][j] - 1/2
        theta_u = X[m-1][j] + 1/2

        # Sum of D[i] over all i for which Y[i] = 1
        val = sum([D[i] if Y[i] == 1 else 0 for i in range(m)])
        valc = 1 - val

        if val < objective_val:
            objective_val = val
            objective_theta = theta_l
            objective_index = j
            objective_b = 1
        if valc < objective_val:
            objective_val = valc
            objective_theta = theta_l
            objective_index = j
            objective_b = -1

        for i in range(m - 1):
            val -= Y[i] * D[i]
            valc += Y[i] * D[i]
            if val < objective_val and X[i][j] < X[i+1][j]:
                objective_val = val
                objective_theta = 1/2 * (X[i][j] + X[i+1][j])
                objective_index = j
                objective_b = 1
            if valc < objective_val and X[i][j] < X[i+1][j]:
                objective_val = valc
                objective_theta = 1/2 * (X[i][j] + X[i+1][j])
                objective_index = j
                objective_b = -1

        val -= Y[m - 1] * D[m - 1]
        valc += Y[m - 1] * D[m - 1]
        if val < objective_val:
            objective_val = val
            objective_theta = theta_u
            objective_index = j
            objective_b = 1
        if val < objective_val:
            objective_val = valc
            objective_theta = theta_u
            objective_index = j
            objective_b = -1

    return objective_index, objective_theta, objective_b

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
