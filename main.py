import matplotlib.pyplot as plt
from utils.dataset import load_data

from layers.convolution import ConvLayer
from layers.activations import ReLU
from layers.pooling import MaxPool
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from tabulate import tabulate
from utils.loss import *

import numpy as np


# Load dataset
X_train, y_train, X_test, y_test = load_data()


# Initialize layers
conv = ConvLayer(
    num_filters=16,
    filter_size=3
)

relu = ReLU()

pool = MaxPool(size=2)

flatten = Flatten()

fc = FullyConnected(
    input_size=2704,
    output_size=10
)


# Forward pass
def forward(image):

    output = conv.forward(image)

    output = relu.forward(output)

    output = pool.forward(output)

    output = flatten.forward(output)

    output = fc.forward(output)

    probabilities = softmax(output)

    return probabilities


# Training parameters
learning_rate = 0.005
epochs = 5

loss_history = []
accuracy_history = []
training_results = []

# Training loop
for epoch in range(epochs):

    total_loss = 0
    correct = 0

    for i in range(1000):

        image = X_train[i][0]

        label = y_train[i]

        # Forward pass
        prediction = forward(image)

        # Loss
        loss = cross_entropy(
            prediction,
            label
        )

        total_loss += loss

        # Accuracy
        if np.argmax(prediction) == label:
            correct += 1

        # Backpropagation
        grad = cross_entropy_gradient(
            prediction,
            label
        )

        grad = fc.backward(
            grad,
            learning_rate
        )

        grad = flatten.backward(grad)

        grad = pool.backward(grad)

        grad = relu.backward(grad)

        conv.backward(
            grad,
            learning_rate
        )

    epoch_loss = total_loss / 1000
    epoch_accuracy = correct / 1000

    training_results.append([
    epoch + 1,
    round(epoch_loss, 4),
    round(epoch_accuracy * 100, 2)
])
    
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)

print(
    f"Epoch {epoch+1} | "
    f"Loss: {epoch_loss:.4f} | "
    f"Accuracy: {epoch_accuracy:.4f}"
)
print("\nTraining Summary:\n")

print(
    tabulate(
        training_results,
        headers=[
            "Epoch",
            "Loss",
            "Accuracy (%)"
        ],
        tablefmt="grid"
    )
)

# Testing
correct = 0

for i in range(1000):

    image = X_test[i][0]

    label = y_test[i]

    prediction = forward(image)

    if np.argmax(prediction) == label:
        correct += 1

print("\nTest Accuracy:", correct / 1000)

# =========================
# LOSS GRAPH
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    marker='o'
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.xticks(range(1, epochs + 1))

plt.grid(True)

plt.savefig("results/loss_graph.png")

plt.show()


# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    range(1, epochs + 1),
    accuracy_history,
    marker='o'
)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.xticks(range(1, epochs + 1))

plt.grid(True)

plt.savefig("results/accuracy_graph.png")

plt.show()