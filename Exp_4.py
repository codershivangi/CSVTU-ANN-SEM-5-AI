# To implement classification using Back Propagation.

import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR problem)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Expected output
y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights and biases randomly
np.random.seed(42)
W1 = np.random.uniform(size=(2,2))
b1 = np.random.uniform(size=(1,2))
W2 = np.random.uniform(size=(2,1))
b2 = np.random.uniform(size=(1,1))

# Learning rate and epochs
lr = 0.5
epochs = 10000

# Training using Backpropagation 

for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    
    # Calculate error
    error = y - final_output

    # Backward pass
    d_output = error + sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # Print error at intervals
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Final trained output
print("Final Output after training:")
print(final_output)