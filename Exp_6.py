# To implement classification of linearly separable Data with a perceptron.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def step(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
  
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.step(np.dot(xi, self.w) + self.b)
                update = self.lr * (target - output)
                self.w += update * xi
                self.b += update
    def predict(self, X):
        return self.step(np.dot(X, self.w) + self.b)
    
# Generate simple linearly separable data
def data():
    X1 = np.random.randn(100,2) + np.array([2,2])
    X2 = np.random.randn(100,2) + np.array([-2,-2])
    X = np.vstack((X1,X2))
    y = np.hstack((np.zeros(100), np.ones(100)))
    return X,y

X,y = data()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

p = Perceptron()
p.fit(X_train,y_train)

print("Weights:",p.w)
print("Bias:",p.b)
print("Accuracy:", np.mean(p.predict(X_test)==y_test))

