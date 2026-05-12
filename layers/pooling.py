import numpy as np


class MaxPool:

    def __init__(self, size):

        self.size = size

    def forward(self, input):

        self.input = input

        num_filters, h, w = input.shape

        output_dim = h // self.size

        output = np.zeros(
            (num_filters, output_dim, output_dim)
        )

        for f in range(num_filters):

            for i in range(output_dim):
                for j in range(output_dim):

                    region = input[
                        f,
                        i*self.size:(i+1)*self.size,
                        j*self.size:(j+1)*self.size
                    ]

                    output[f, i, j] = np.max(region)

        return output

    def backward(self, grad_output):

        num_filters, h, w = self.input.shape

        grad_input = np.zeros(self.input.shape)

        output_dim = h // self.size

        for f in range(num_filters):

            for i in range(output_dim):
                for j in range(output_dim):

                    region = self.input[
                        f,
                        i*self.size:(i+1)*self.size,
                        j*self.size:(j+1)*self.size
                    ]

                    max_val = np.max(region)

                    for x in range(self.size):
                        for y in range(self.size):

                            if region[x, y] == max_val:

                                grad_input[
                                    f,
                                    i*self.size+x,
                                    j*self.size+y
                                ] = grad_output[f, i, j]

        return grad_input
# =========================
# FILE: layers/flatten.py
# =========================

class Flatten:

    def forward(self, input):

        self.input_shape = input.shape

        return input.flatten()

    def backward(self, grad_output):

        return grad_output.reshape(self.input_shape)