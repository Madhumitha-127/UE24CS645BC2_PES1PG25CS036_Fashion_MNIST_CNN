import numpy as np


class ConvLayer:

    def __init__(self, num_filters, filter_size):

        self.num_filters = num_filters
        self.filter_size = filter_size

        self.filters = np.random.randn(
            num_filters,
            filter_size,
            filter_size
        ) / (filter_size * filter_size)

    def forward(self, input):

        self.input = input

        h, w = input.shape

        output_dim = h - self.filter_size + 1

        output = np.zeros(
            (self.num_filters, output_dim, output_dim)
        )

        for f in range(self.num_filters):

            current_filter = self.filters[f]

            for i in range(output_dim):
                for j in range(output_dim):

                    region = input[
                        i:i+self.filter_size,
                        j:j+self.filter_size
                    ]

                    output[f, i, j] = np.sum(
                        region * current_filter
                    )

        return output

    def backward(self, grad_output, learning_rate):

        grad_filters = np.zeros(self.filters.shape)

        for f in range(self.num_filters):

            for i in range(grad_output.shape[1]):
                for j in range(grad_output.shape[2]):

                    region = self.input[
                        i:i+self.filter_size,
                        j:j+self.filter_size
                    ]

                    grad_filters[f] += (
                        grad_output[f, i, j] * region
                    )

        self.filters -= learning_rate * grad_filters