from math import log

def core_gain(X, Y, labels, i, cfunc):
    if len(X) == 0 or len(labels) == 0:
        raise "Gain function called with zero length array"
    if len(X) != len(Y):
        raise "Gain function called with different size arrays"
    if i >= len(X[0]):
        raise "Index exceeds array bounds in gain function call"

    
