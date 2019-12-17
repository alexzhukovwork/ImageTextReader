import numpy as np
import NN.load_data
import copy
from matplotlib import pyplot as plt
# Model setup
from numba import cuda, float32, float64
import math

import time


def initialize_weights(n_x, n_h):
    w = np.random.randn(n_h, n_x) * xavier_initialization(n_x)
    b = np.zeros((n_h, 1), dtype=np.float32)
    return (w, b)


def xavier_initialization(n_x):
    return 2 / n_x


# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 8


@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    #    print(A.shape[1])
    #   print(A.shape[1] / TPB)
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


def linear(w, x, b):
    # (n_h, n_x) * (n_x, m) + (n_h, 1) = (n_h, m)
    # Copy the arrays to the device
    # The data array
    #   A = w #.astype(float)  # [32 x 48] matrix containing all 3's
    #    B = x #.astype(float)  # [48 x 16] matrix containing all 4's


    return np.dot(w, x) + b

'''

    A_global_mem = cuda.to_device(w)
    B_global_mem = cuda.to_device(x)
    C_global_mem = cuda.device_array((w.shape[0], x.shape[1]))  # [32 x 16] matrix result

  #  print(str(w.shape[0]) + " " + str(x.shape[1]))



    # Configure the blocks
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(w.shape[0] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(x.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Start the kernel
    fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    res = C_global_mem.copy_to_host()

 #   print(res)
#    print(str(np.dot(w, x)))
#
   # print(C)
'''



def linear_d(dz, w, a_prev, b):
    # b = (n_h, 1)
    # w = (n_h, n_x)
    # dz = (n_h, m)
    # a = (n_x, m)
    _, m = a_prev.shape

    da_prev = np.dot(w.T, dz)  # (n_x, n_h) * (n_h, m)
    dw = 1 / m * np.dot(dz, a_prev.T)  # (n_h, m) * (m, n_x)
    db = np.mean(dz, axis=1, keepdims=True)  # (n_h, m) / m
    return da_prev, dw, db


def relu(z):
    return np.maximum(z, 0)


def relu_d(a):
    return np.int64(a > 0)


def softmax(z):
    # Shift z values so highest value is 0
    # Must stabilize as exp can get out of control
    z_norm = z - np.max(z)
    exp = np.exp(z_norm)
    return exp / np.sum(exp, axis=0, keepdims=True)


def softmax_d(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


# def softmax_d(z):
#     Sz = softmax(z)
#     D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
#     return D


# def softmax_d_m(z):
#     # Remember to reverse n_m and n_class
#     zt = z.T
#     n_m, n_class = zt.shape
#     s_grad = np.empty((n_m, n_class, n_class))
#     for i in range(zt.shape[0]):
#         row = zt[i]
#         soft_grad = softmax_d(row)
#         s_grad[i] = soft_grad
#     return s_grad.T

def softmax_d_m(z):
    # Remember to reverse n_m and n_class
    n_class, n_m = z.shape
    s_grad = np.empty((n_class, n_class, n_m))
    for i in range(z.shape[1]):
        row = z[:, i]
        soft_grad = softmax_d(row)
        s_grad[:, :, i] = soft_grad
    return s_grad


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoid_d(z):
    # a must be sigmoid activated
    return z * (1 - z)


#
def compute_cost(Y, z):
    return - np.mean(Y * np.log(z) + (1 - Y) * np.log(1 - z))


def categorical_cross_entropy(y, a):
    cost = np.sum(y * np.log(a), axis=1, keepdims=True)
    return - np.mean(cost)


def categorical_cross_entropy_d(y, a3):
    # cost_d = y / a3 + (1 - y) / (1 - a3)
    # return - cost_d
    return - (y / a3)


def binary_cross_entropy(y, a):
    cost = y * np.log(a) + (1 - y) * np.log(1 - a)
    return - np.mean(cost)


def binary_cross_entropy_d(y, a):
    # cost_d = y / a + (1 - y) / (1 - a)
    cost_d = y - a / (y * (1 - y))  # same as above
    return - cost_d


def forward_pass(X, Y, weights):
    w1, b1, w2, b2, w3, b3 = weights
    # forward pass
    z1 = linear(w1, X, b1)
    a1 = relu(z1)

    z2 = linear(w2, a1, b2)
    a2 = relu(z2)

    z3 = linear(w3, a2, b3)
    a3 = softmax(z3)

    # Cost
    cost = categorical_cross_entropy(Y, a3)
    return (cost, (z1, a1, z2, a2, z3, a3))


def forward_pass_check(X, weights):
    w1, b1, w2, b2, w3, b3 = weights
    # forward pass
    z1 = linear(w1, X, b1)
    a1 = relu(z1)

    z2 = linear(w2, a1, b2)
    a2 = relu(z2)

    z3 = linear(w3, a2, b3)
    a3 = softmax(z3)

    return (z1, a1, z2, a2, z3, a3)


def backpropagate(X, Y, weights, activations):
    w1, b1, w2, b2, w3, b3 = weights
    z1, a1, z2, a2, z3, a3 = activations

    dz3 = a3 - Y

    cost_d = categorical_cross_entropy_d(Y, a3)
    a3_d = softmax_d_m(a3)
    #   print('A3', a3.shape)
    #  print(cost_d.shape)
    #   print(a3_d.shape)
    cost_d_r = cost_d.reshape((cost_d.shape[0], 1, cost_d.shape[1]))
    dz3_step = np.einsum('ijk,jyk->iyk', a3_d, cost_d_r)
    dz3_step_r = dz3_step.reshape((dz3_step.shape[0], dz3_step.shape[2]))

    dz3_test = np.einsum('ijk,jk->ik', a3_d, cost_d)

    da2, dw3, db3 = linear_d(dz3, w3, a2, b3)
    dz2 = relu_d(a2) * da2

    da1, dw2, db2 = linear_d(dz2, w2, a1, b2)
    dz1 = relu_d(a1) * da1

    _, dw1, db1 = linear_d(dz1, w1, X, b1)
    return dw1, db1, dw2, db2, dw3, db3


# Let's create a model with 2 hidden layers with 100 units


def check(X_test, weights):
    activations = forward_pass_check(X_test, weights=weights)
    z1, a1, z2, a2, z3, a3 = activations

    pred = np.zeros(a3.shape)
    pred[a3.argmax(axis=0), np.arange(a3.shape[1])] = 1

    return pred


def gradient_check(X, Y):
    n_x, n_m = X.shape
    # n_y, _ = Y_train.shape
    n_y = 1
    n_h1, n_h2 = [1024, 1024]

    w1, b1 = initialize_weights(n_x, n_h1)
    w2, b2 = initialize_weights(n_h1, n_h2)
    w3, b3 = initialize_weights(n_h2, n_y)

    weights = w1, b1, w2, b2, w3, b3
    cost1, activations = forward_pass(X, Y, weights)
    gradients = backpropagate(X, Y, weights, activations)
    approx_gradients = copy.deepcopy(gradients)

    # Gradient checking
    epsilon = .00001
    all_weights = (w1, b1, w2, b2, w3, b3)
    num_parameters = len(all_weights)

    for i in range(num_parameters):
        current_param = all_weights[i]

        for row in range(current_param.shape[0]):
            for col in range(current_param.shape[1]):
                thetaplus = copy.deepcopy(all_weights)
                thetaminus = copy.deepcopy(all_weights)

                thetaplus[i][row, col] = (thetaplus[i][row, col] + epsilon)
                thetaminus[i][row, col] = (thetaminus[i][row, col] - epsilon)

                J_plus, _ = forward_pass(X, Y, thetaplus)
                J_minus, _ = forward_pass(X, Y, thetaminus)

                approx = (J_plus - J_minus) / (2 * epsilon)
                approx_gradients[i][row, col] = approx
        print('Completed param:', i)

    def euclidean(x):
        return np.sqrt(np.sum(x ** 2))

    def flat_array(x):
        res = np.array([])
        for i in range(len(x)):
            res = np.concatenate((res, x[i].flatten()))
        return res

    np_gradients = flat_array(gradients)
    np_gradients_approx = flat_array(approx_gradients)
    numerator = euclidean(np.array(np_gradients) - np.array(np_gradients_approx))
    denominator = euclidean(np_gradients) + euclidean(np_gradients_approx)
    difference = numerator / denominator
    return difference


def model(X_train, Y_train, num_iterations=50, learning_rate=0.01):
    start_time = time.time()
    n_x, n_m = X_train.shape
    n_y, _ = Y_train.shape

    n_h1, n_h2 = [1024, 1024]

    w1, b1 = initialize_weights(n_x, n_h1)
    w2, b2 = initialize_weights(n_h1, n_h2)
    w3, b3 = initialize_weights(n_h2, n_y)

    for i in range(num_iterations):
        # forward pass
        weights = w1, b1, w2, b2, w3, b3
        cost, activations = forward_pass(X_train, Y_train, weights)
        #   print('Cost:', cost)

        gradients = backpropagate(X_train, Y_train, weights, activations)
        dw1, db1, dw2, db2, dw3, db3 = gradients

        assert (dw3.shape == w3.shape)
        assert (dw2.shape == w2.shape)
        assert (dw1.shape == w1.shape)

        # Update weights
        w3 -= learning_rate * dw3
        b3 -= learning_rate * db3
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

    # Accuracy
    weights = w1, b1, w2, b2, w3, b3

    print("time: " + str(time.time() - start_time))

    return weights
