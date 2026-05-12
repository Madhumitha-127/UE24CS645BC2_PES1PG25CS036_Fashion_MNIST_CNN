
import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size):

        self.weights = np.random.randn(
            input_size,
            output_size
        ) * np.sqrt(2 / input_size)

        self.biases = np.zeros(output_size)

    def forward(self, input):

        self.input = input

        return np.dot(input, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):

        grad_weights = np.outer(
            self.input,
            grad_output
        )

        grad_biases = grad_output

        grad_input = np.dot(
            self.weights,
            grad_output
        )

        self.weights -= learning_rate * grad_weights

        self.biases -= learning_rate * grad_biases

        return grad_input