from math import log

def train_error_gain(a):
    return min(a, 1 - a)

def information_gain(a):
    return 0 if a < 1e-10 or a > 1-1e-10 else -a * log(a) - (1-a) * log(1-a)

def gini_index_gain(a):
    return 2 * a * (1 - a)

def calc_gain(xp, xn, yp, yp_and_xp, gain_measure=information_gain):
    """
    Calculate
        C(P[y=1]) - (P[x=1] * C(P[y=1 | x=1]) + P[x=0] * C(P[y=1 | x=0]))
    where we have replaced x=1 and x=0 by xp and xn to allow real-valued input
    Inputs:
        xp - count of positive x
        xn - count of negative x
        yp - count of positive y (usually y = 1)
        yp_and_xp - count of points with positive x and positive y
        gain_measure - wanted gain measure as a function
    """
    Pxp = xp / (xp + xn)
    Pxn = 1 - Pxp
    Pyp_g_xp = yp_and_xp / xp if xp > 0 else 0
    Pyp_g_xn = (yp - yp_and_xp) / xn if xn > 0 else 0
    Pyp = yp / (xp + xn)
    return gain_measure(Pyp) \
            - Pxp * gain_measure(Pyp_g_xp) \
                - Pxn * gain_measure(Pyp_g_xn)
