# UE24CS645BC2_PES1PG25CS036_Fashion_MNIST_CNN

# Fashion MNIST CNN From Scratch

## Project Overview

This project implements a Convolutional Neural Network (CNN) completely from scratch using Python and NumPy for image classification on the Fashion-MNIST dataset.

The project manually implements:
- Convolution Layer
- ReLU Activation
- Max Pooling
- Flatten Layer
- Fully Connected Layer
- Forward Propagation
- Backpropagation

---

# Dataset

The Fashion-MNIST dataset contains grayscale images of clothing items belonging to 10 classes.

## Dataset Details

| Property | Value |
|---|---|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 |
| Number of Classes | 10 |

---

# CNN Architecture

```text
Input Image (28x28)
        ↓
Convolution Layer (16 filters, 3x3)
        ↓
ReLU Activation
        ↓
Max Pooling Layer (2x2)
        ↓
Flatten Layer
        ↓
Fully Connected Layer
        ↓
Softmax Output Layer
```

---

# Project Structure

```text
UE24CS645BC2_PES1PG25CS036_Fashion_MNIST_CNN/
│
├── main.py
├── README.md
│
├── layers/
│   ├── convolution.py
│   ├── pooling.py
│   ├── flatten.py
│   ├── fully_connected.py
│   └── activations.py
│
├── utils/
│   ├── dataset.py
│   └── loss.py
│
└── results/
    ├── accuracy_graph.png
    └── loss_graph.png
```

---

# Results

| Metric | Value |
|---|---|
| Training Accuracy | 84.9% |
| Test Accuracy | 76.2% |

---

# Output Graphs

The project generates:
- Training Loss Graph
- Training Accuracy Graph

Graphs are saved inside the `results/` folder.

---

# How to Run

## Clone Repository

```bash
git clone https://github.com/Madhumitha-127/UE24CS645BC2_PES1PG25CS036_Fashion_MNIST_CNN.git
```

## Open Project Folder

```bash
cd UE24CS645BC2_PES1PG25CS036_Fashion_MNIST_CNN
```

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Virtual Environment

### Windows

```bash
.\.venv\Scripts\Activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install numpy matplotlib tensorflow tabulate
```

## Run Project

```bash
python main.py
```

---

# Learning Outcomes

This project helped in understanding:
- CNN architecture
- Convolution operation
- Feature extraction
- Pooling
- Backpropagation
- Gradient descent
- Image classification

---

# Conclusion

This project demonstrates the implementation of a Convolutional Neural Network from scratch using NumPy without relying on high-level deep learning libraries.