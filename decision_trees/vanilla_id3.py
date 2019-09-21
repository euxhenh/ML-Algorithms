import sys

class Node:
    def __init__(self, label):
        self.label = label
        self.yesbranch = None
        self.nobranch = None

def gain(X, Y, i):
    assert(len(X) > 0)
    x0, x1 = 0, 0
    y1x0, y1x1, = 0, 0

    for x, y in zip(X, Y):
        if x[i] == 1:
            x1 += 1
            if y == 1:
                y1x1 += 1
        else:
            x0 += 1
            if y == 1:
                y1x0 += 1

    py1 = (y1x0 + y1x1) / len(X)
    Cpy1 = min(py1, 1 - py1)
    
    if x0 != 0:
        py1x0 = y1x0 / x0
    if x1 != 0:
        py1x1 = y1x1 / x1

    px0py1x0 = x0 / len(X) * min(py1x0, 1 - py1x0)
    px1py1x1 = x1 / len(X) * min(py1x1, 1 - py1x1)
    
    return Cpy1 - px0py1x0 - px1py1x1

def id3(X, Y, A):
    counter = 0
    for y in Y:
        if y == 1:
            counter += 1

    if len(A) == 0 or counter == 0 or counter == len(Y):
        if counter >= len(Y)/2:
            return Node(1)
        else:
            return Node(-1)
        return Node(1 if counter >= len(Y)/2 else -1)

    j = -1
    G = -float("inf")
    for i in A:
        g = gain(X, Y, i)
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

    Tno = id3(noX, noY, noA)
    Tyes = id3(yesX, yesY, yesA)

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

    tree = id3(X, Y, A)

    for x, y in zip(X, Y):
        if infer(tree, x) != y:
            print('nooo')

if __name__ == '__main__':
    call_id3()
