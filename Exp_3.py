# To perform the basic matrix operations.

import numpy as np

# Input matrices
A = np.array([[1, 2, 3],[4, 5, 6]])
B = np.array([[7, 8, 9],[10, 11, 12]])

# Addition
print("A + B:\n", A + B)

# Subtraction
print("A - B:\n", A - B)

# Element-wise Multiplication
print("A * B (Element-wise):\n", A * B)

# Matrix Multiplication (if dimensions allow)
if A.shape[1] == B.shape[0]:
  print("A x B (Matrix Multiplication):\n", np.dot(A, B))
else:
  print("Matrix multiplication not possible.")

# Transpose
print("Transpose of A:\n", A.T)
print("Transpose of B:\n", B.T)