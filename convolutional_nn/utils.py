import numpy as np


def random_weight_init(input, output):
    b = np.sqrt(6) / np.sqrt(input + output)
    return np.random.uniform(-b, b, (input, output))


def one_bias_init(outd):
    return np.ones((outd, 1))


def random_weight_init_conv(k_num, C, k_height, k_width):
    b = np.sqrt(6 / ((k_num + C) * k_height * k_width))
    return np.random.uniform(-b, b, (k_num, C, k_height, k_width))


def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    Input:
    X: 4D tensor of shape [N, C, H, W], intput feature map
    k_height, k_width: height and width of convolution kernel
    Output:
    cols: 2D array
    '''
    N, C, H, W = X.shape
    bigM = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                  mode='constant', constant_values=0)
    cols = []
    Hs = H + 2*padding - k_height + 1
    Ws = W + 2*padding - k_width + 1

    for j in range(0, Hs, stride):
        for i in range(0, Ws, stride):
            cols.append(bigM[:, :, j:j+k_height, i:i+k_width].reshape(N, -1))
    return np.vstack(cols).T


def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    Input:
    grad_X_col: 2D array
    X_shape: (N, C, H, W)
    k_height, k_width: height and width of convolution kernel
    Output:
    X_grad: 4D tensor of shape X_shape, gradient w.r.t. feature map
    '''
    N, C, H, W = X_shape
    grad_X_col = grad_X_col.T
    bigM = np.zeros(shape=(N, C, H + 2*padding, W + 2*padding))
    Hs = 1 + (H - k_height + 2*padding) // stride
    Ws = 1 + (W - k_width + 2*padding) // stride

    for j in range(Hs):
        for i in range(Ws):
            block = grad_X_col[(j*Ws*N + i*N): (j*Ws*N + i*N + N), :]
            block = block.reshape(N, C, k_height, k_width)
            bigM[:, :, j: j + k_height, i: i + k_width] += block
    return bigM[:, :, padding:-padding, padding:-padding]
