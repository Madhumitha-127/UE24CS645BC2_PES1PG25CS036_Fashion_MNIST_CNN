from tensorflow.keras.datasets import fashion_mnist
import numpy as np

def load_data():

    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Add channel dimension
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, y_train, X_test, y_test