class Flatten:

    def forward(self, input):

        self.input_shape = input.shape

        return input.flatten()

    def backward(self, grad_output):

        return grad_output.reshape(self.input_shape)