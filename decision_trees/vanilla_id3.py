import sys
from gain_functions import train_error_gain, information_gain, gini_index_gain

DEPTH = 8

class Node:
    def __init__(self, label):
        self.label = label
        self.yesbranch = None
        self.nobranch = None

    def isleaf(self):
        if self.yesbranch == None and self.nobranch == None:
            return True
        else:
            return False

def id3(X, Y, A, depth):
    counter = 0
    for y in Y:
        if y == 1:
            counter += 1

    if len(A) == 0 or counter == 0 or counter == len(Y):
        return Node(1 if counter >= len(Y)/2 else -1)
    
    if depth > DEPTH:
        return Node(1 if counter >= len(Y)/2 else -1)

    j = -1
    G = -float("inf")
    for i in A:
        g = gini_index_gain(X, Y, i)
        if G < g:
            G, j = g, i

    noX, noY = [], []
    yesX, yesY = [], []

    for x, y in zip(X, Y):
        if x[j] == 1:
            yesX.append(x)
            yesY.append(y)
        else:
            noX.append(x)
            noY.append(y)

    noA, yesA = A.copy(), A.copy()
    noA.remove(j)
    yesA.remove(j)

    Tno = id3(noX, noY, noA, depth+1)
    Tyes = id3(yesX, yesY, yesA, depth+1)

    node = Node(j)
    node.yesbranch = Tyes
    node.nobranch = Tno
    return node

def infer(tree, x):
    if tree.nobranch == None and tree.yesbranch == None:
        return tree.label

    if x[tree.label] == 1:
        return infer(tree.yesbranch, x)
    else:
        return infer(tree.nobranch, x)

def call_id3():
    fl = open("data", "r")
    X = []
    Y = []
    lines = fl.readlines()
    for line in lines:
        x = [int(x) for x in line.strip().split(' ')]
        y = x[-1]
        del x[-1]

        X.append(x)
        Y.append(y)

    A = list(range(len(X[0])))

    tree = id3(X, Y, A, depth=1)

    errors = 0
    for x, y in zip(X, Y):
        if infer(tree, x) != y:
            errors += 1
    print("Sample Accuracy: ", 1-errors/len(X))

if __name__ == '__main__':
    call_id3()
