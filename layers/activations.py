import numpy as np


class ReLU:

    def forward(self, input):

        self.input = input

        return np.maximum(0, input)

    def backward(self, grad_output):

        grad = grad_output.copy()

        grad[self.input <= 0] = 0

        return grad