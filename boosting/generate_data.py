import random

def generate_labels(X):
    Y = []
    d = len(X[0])
    for x in X:
        if x[d-1] > 7:
            Y.append(1)
        else:
            Y.append(-1)
    return Y

def generate_data_points(m, d):
    X = []
    for i in range(m):
        x = []
        for j in range(d):
            x.append(random.uniform(0, 10))
        X.append(x)
    return X

def write_sample_set_to_file(X, Y, name="data"):
    fl = open(name, "w")
    for x, y in zip(X, Y):
        for feat in x:
            fl.write(str(feat) + " ")
        fl.write(str(y) + "\n")
    fl.close()

if __name__ == '__main__':
    X = generate_data_points(1000, 4)
    Y = generate_labels(X)
    write_sample_set_to_file(X, Y)
