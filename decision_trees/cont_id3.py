from math import log
import gains

YES = 1
NO = -1

class DecisionNode:
    def __init__(self, label, theta=0.5):
        self.label = label
        self.theta = theta
        self.yesbranch = None
        self.nobranch = None

    def infer(self, x):
        if self.isleaf():
            return self.label
        elif x[self.label] - self.theta > 0:
            return self.yesbranch.infer(x)
        else:
            return self.nobranch.infer(x)

    def isleaf(self):
        if self.yesbranch == None and self.nobranch == None:
            return True
        return False

def id3(X, Y, A, cfunc=gains.information_gain):
    """
    X - Feature vectors
    Y - Labels for X
    A - Subset of [len(X[0])]
    """
    m = len(X)
    ypos = Y.count(YES)

    # We have reached the end
    if len(A) == 0 or ypos == len(Y) or ypos == 0:
        return DecisionNode(YES if ypos >= len(Y)/2 else NO)

    max_gain = -1                    # Will try to maximize
    best_feature = 0
    theta = 0.5

    # We want to find a feature which maximizes gain
    for feature in A:
        # Sort using this feature
        X, Y = zip(*sorted(zip(X, Y), key=lambda row: row[0][feature]))
        
        # Theta is smaller than all x
        ypos_and_xpos = ypos
        g = gains.calc_gain(m, 0, ypos, ypos_and_xpos, cfunc)
        if g > max_gain:
            max_gain, best_feature, theta = g, feature, X[0][feature] - 1/2
        
        # Theta is midway along x_i and x_{i+1}
        # At each step we move one x_i on the other side of the wall
        for i in range(m - 1):
            if Y[i] == YES:   ypos_and_xpos -= 1
            g = gains.calc_gain(m - i - 1, i + 1, ypos, ypos_and_xpos, cfunc)

            # Update
            if g > max_gain and X[i][feature] != X[i+1][feature]:
                max_gain, best_feature = g, feature
                theta = (X[i][feature] + X[i+1][feature]) / 2

        # Theta is greater than all x
        if Y[m - 1] == YES:   ypos_and_xpos -= 1
        g = gains.calc_gain(0, m, ypos, ypos_and_xpos, cfunc)
        if g > max_gain:
            max_gain, best_feature, theta = g, feature, X[m-1][feature] + 1/2

    # Split the dataset into two groups on the basis of the sign of x[j] - theta
    X_neg, X_pos = [], []
    Y_neg, Y_pos = [], []

    for x, y in zip(X, Y):
        if x[best_feature] - theta > 0:
            X_pos.append(x)
            Y_pos.append(y)
        else:
            X_neg.append(x)
            Y_neg.append(y)

    # Remove the best feature found from feature set
    A_no_j_neg = A.copy()
    A_no_j_neg.remove(best_feature)
    A_no_j_pos = A_no_j_neg.copy()

    # Recur on both branches
    T_pos = id3(X_pos, Y_pos, A_no_j_pos)
    T_neg = id3(X_neg, Y_neg, A_no_j_neg)

    # Construct tree
    node = DecisionNode(best_feature, theta)
    node.yesbranch = T_pos
    node.nobranch = T_neg
    return node

def call_id3():
    fl = open("diabetes.csv", "r")
    X = []
    Y = []
    lines = fl.readlines()
    for line in lines[1:]:
        x = [float(x) for x in line.strip().split(',')]
        y = -1 if int(x[-1]) == 0 else 1
        del x[-1]

        X.append(x)
        Y.append(y)

    A = list(range(len(X[0])))

    tree = id3(X, Y, A, cfunc=gains.gini_index_gain)

    errors = 0
    for x, y in zip(X, Y):
        if tree.infer(x) != y:
            errors += 1
    print("Sample Accuracy: ", 1-errors/len(X))

if __name__ == '__main__':
    call_id3()
