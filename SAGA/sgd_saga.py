import numpy as np
import pandas as pd
import cvxpy as cp
from matplotlib import pyplot as plt

def f(X, y, w):
    return 1/X.shape[0] * np.sum(np.log(1 + np.exp(-y * (X@w))))

def cvxpy_solution(X, y):
    N = X.shape[0]
    d = X.shape[1]
    w = cp.Variable((d,1))
    objective = cp.Minimize(1/N * cp.sum(cp.logistic(cp.multiply(-y, X@w))))
    prob = cp.Problem(objective)
    result = prob.solve(solver='SCS')
    return w.value, result

################# SGD ###################
def fit(X, y, batch_size, lr, iters=500):
    fvals = []
    w = np.zeros((X.shape[1], 1))

    for _ in range(iters):
        batch_indices = np.random.randint(0, X.shape[0], batch_size)
        batch_x = X[batch_indices]
        batch_y = y[batch_indices]

        e = np.exp(batch_y * (batch_x @ w))
        df = np.mean(-batch_y * batch_x / (1 + e), axis=0).reshape(-1, 1)
        w -= lr * df
        fvals.append(f(X, y, w))

    return np.array(fvals)

def mean_fit(X, y, batch_size, lr, iters=500, runs=25):
    mean_fvals = np.zeros((iters,))
    for _ in range(runs):
        mean_fvals += fit(X, y, batch_size, lr, iters)
    return mean_fvals / runs

def sgd():
    samples = pd.read_csv('samples.csv', header=None).to_numpy()
    X = samples[:, :100]
    y = samples[:, 100]
    y = y.reshape((-1, 1))

    w_sol, f_sol = cvxpy_solution(X, y)
    print(f'Optimal function value is {f_sol}.')

    batch_list = [1, 10, 100, 1000]
    lr_list = [1, 0.3, 0.1, 0.03]

    d = {}

    for lr in lr_list:
        for batch_size in batch_list:
            mean_fvals = mean_fit(X, y, batch_size, lr)
            print(f'Final average value for b={batch_size}, eta={lr} :: {mean_fvals[-1]}')
            d[(lr, batch_size)] = mean_fvals.copy()

    # Fix eta
    for lr in lr_list:
        for batch_size in batch_list:
            plt.plot(np.log(d[(lr, batch_size)] - f_sol), label=f'b={batch_size}')
        plt.title(f'eta={lr}')
        plt.xlabel('Iteration')
        plt.ylabel('log(fhat_t - f^*)')
        plt.legend()
        plt.show()

    # Fix b
    for batch_size in batch_list:
        for lr in lr_list:
            plt.plot(np.log(d[(lr, batch_size)] - f_sol), label=f'eta={lr}')
        plt.title(f'b={batch_size}')
        plt.xlabel('Iteration')
        plt.ylabel('log(fhat_t - f^*)')
        plt.legend()
        plt.show()

################ SAGA ###################
def saga_fit(X, y, batch_size, lr, iters=500):
    fvals = []
    N = X.shape[0]
    w = np.zeros((X.shape[1], 1))
    g = -y * X / 2 # since w = 0
    g_mean = np.mean(g, axis=0).reshape(-1, 1)

    for _ in range(iters):
        batch_indices = np.random.randint(0, X.shape[0], batch_size)
        batch_x = X[batch_indices]
        batch_y = y[batch_indices]
        batch_g = g[batch_indices]

        e = np.exp(batch_y * (batch_x @ w))
        df = -batch_y * batch_x / (1 + e)
        df_mean = np.mean(df, axis=0).reshape(-1, 1)
        g_i_mean = np.mean(batch_g, axis=0).reshape(-1, 1)
        w -= lr * (df_mean - g_i_mean + g_mean)

        # Update g[i] and g_mean
        g_mean = g_mean - batch_size/N * (g_i_mean - df_mean)
        g[batch_indices] = df.copy()

        fvals.append(f(X, y, w))

    return np.array(fvals)

def saga_mean_fit(X, y, batch_size, lr, iters=500, runs=25):
    mean_fvals = np.zeros((iters,))
    for _ in range(runs):
        mean_fvals += saga_fit(X, y, batch_size, lr, iters)
    return mean_fvals / runs

def saga():
    samples = pd.read_csv('samples.csv', header=None).to_numpy()
    X = samples[:, :100]
    y = samples[:, 100]
    y = y.reshape((-1, 1))

    w_sol, f_sol = cvxpy_solution(X, y)
    print(f'Optimal function value is {f_sol}.')

    lr = 0.1
    batch_list = [1, 10, 100]

    for batch_size in batch_list:
        saga_mean_fvals = saga_mean_fit(X, y, batch_size, lr)
        print(f'Final average value for b={batch_size}, eta={lr}' +\
                f' :: {saga_mean_fvals[-1]}')
        mean_fvals = mean_fit(X, y, batch_size, lr)
        plt.plot(np.log(mean_fvals - f_sol), label='SGD')
        plt.plot(np.log(saga_mean_fvals - f_sol), label='SAGA')
        plt.title(f'b={batch_size}, eta={lr}')
        plt.xlabel('Iteration')
        plt.ylabel('log(fhat - f*)')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    print("############ SGD ############")
    sgd()
    print("############ SAGA ############")
    saga()

