import random
import numpy as np
from matplotlib import pyplot as plt

def sgn(x):
    return 1 if x > 0 else -1

def func(x):
    return 1 - 2 * x

def generate_2d_points(f, num=10, minb=-10, maxb=10):
    """
        Generate points that fall into different sides of function f.
        params:
            f:              Python function
            num:            number of points to generate
            [minb, maxb]:   interval of points
        returns:
            np array with the following header
                ____________________
                | x0 | x1 | x2 | y |
                ____________________
    """
    points = []

    for i in range(num):
        x1 = random.uniform(minb, maxb)
        x2 = random.uniform(minb, maxb)

        if (x2 < f(x1)):
            points.append([1, x1, x2, -1])
        else:
            points.append([1, x1, x2, 1])

    return np.asarray(points)

def plot_points(ax, x, y):
    ax.scatter(x, y)
    return ax

def plot_func(ax, f, minb, maxb):
    """
        Plots a function
        params:
            ax:             np.axes(), the axes where to plot
            f :             a python function
            [minb, maxb]:   interval used for plotting
    """
    x = np.linspace(minb, maxb, 1000)
    ax.plot(x, f(x))
    return ax

def plot_func_param(ax, coeffs, minb, maxb):
    """
        Plots a function in canonical form, with coeffs an array of coefficients
        params:
            coeffs:         np array of coefficients
    """
    x = np.linspace(minb, maxb, 1000)
    y = (-coeffs[0] - coeffs[1]*x) / coeffs[2]
    ax.scatter(x, y)
    return ax

def train():
    """
        Train linear model where the points are assumed to be linearly separable
        The learning algorithm is given by updating the weights w as follows:
            w(t + 1) = w(t) + y(t) * x(t)
        for every missclassified point x, at a given time t, where y is the gt.
    """
    w = []
    bound = 10
    # Randomly initialize the weights
    for i in range(3):
        w.append(random.uniform(-bound, bound))
    w = np.asarray(w)

    # Generate 2D points and plot them
    points = generate_2d_points(func, 100)
    ax = plt.axes()
    ax = plot_points(ax, points[:, 1], points[:, 2])
    ax = plot_func(ax, func, -bound, bound)

    # Train until all points predicted correctly
    while True:
        wrong = False
        for i, row in enumerate(points):
            x, gt = row[:-1], row[-1] # input, ground truth
            y = sgn(np.dot(w, x.transpose()))

            # If missclassified, update weights
            if y != gt:
                w = w + gt * x.transpose()
                wrong = True

        if wrong == False:
            break

    # Plot results
    ax = plot_func_param(ax, w, -bound, bound)
    ax.axis('equal')
    plt.show()

if __name__ == '__main__':
    train()
