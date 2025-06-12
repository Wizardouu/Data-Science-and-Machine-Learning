import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import r2_score
import numpy as np

def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2).item()

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = torch.zeros(n_features, 1, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)
        
        for i in range(self.n_iters):
            y_predicted = torch.mm(X, self.weights) + self.bias
            loss = torch.mean((y_predicted - y) ** 2)
            loss.backward()
            
            with torch.no_grad():
                self.weights -=  self.lr * self.weights.grad
                self.bias -=  self.lr * self.bias.grad
                self.weights.grad.zero_()
                self.bias.grad.zero_()
    
    def predict(self, X):
        with torch.no_grad():
            return torch.mm(X, self.weights) + self.bias

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    model = LinearRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    predictions_np = predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    mse = mean_squared_error(y_test, predictions)
    print(f"MSE: {mse:.4f}")
    
    accu = r2_score(y_test_np, predictions_np)
    print(f"Accuracy (R^2 Score): {accu:.4f}")
    
    with torch.no_grad():
        weight = model.weights.item()
        bias = model.bias.item()
        print(f"Learned weight: {weight:.4f}")
        print(f"Learned bias: {bias:.4f}")
    
    plt.scatter(X_test.cpu().numpy(), y_test_np, color='blue', label='Actual')
    plt.plot(X_test.cpu().numpy(), predictions_np, color='red', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
