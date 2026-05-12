
import numpy as np


def softmax(x):

    exp = np.exp(x - np.max(x))

    return exp / np.sum(exp)


def cross_entropy(prediction, label):

    return -np.log(prediction[label])


def cross_entropy_gradient(prediction, label):

    grad = prediction.copy()

    grad[label] -= 1

    return grad