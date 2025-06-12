import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def hinge_loss(self, y_true, y_pred):
        return torch.mean(torch.clamp(1 - y_true * y_pred, min=0))
    
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y)
            
        y = torch.where(y <= 0, -1, 1).float()
        
        n_samples, n_features = X.shape
        self.w = torch.zeros(n_features, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        
        for i in range(self.n_iters):
            for idx in range(n_samples):
                x_i = X[idx:idx+1]
                y_i = y[idx:idx+1]
                
                distance = y_i * (torch.dot(x_i.flatten(), self.w) - self.b)
                
                if distance >= 1:
                    loss = self.lambda_param * torch.sum(self.w ** 2)
                else:
                    loss = self.lambda_param * torch.sum(self.w ** 2) + (1 - distance)
                
                loss.backward()
                
                with torch.no_grad():
                    self.w -= self.lr * self.w.grad
                    self.b -= self.lr * self.b.grad
                    self.w.grad.zero_()
                    self.b.grad.zero_()
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        with torch.no_grad():
            approx = torch.mm(X, self.w.unsqueeze(1)).flatten() - self.b
            return torch.sign(approx).numpy()
    
    def decision_function(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        with torch.no_grad():
            return (torch.mm(X, self.w.unsqueeze(1)).flatten() - self.b).numpy()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_linear, y_linear = datasets.make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                                     n_informative=2, random_state=1, n_clusters_per_class=1)
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, 
                                                                                   test_size=0.2, random_state=1234)
    
    X_train_torch = torch.FloatTensor(X_train_linear).to(device)
    X_test_torch = torch.FloatTensor(X_test_linear).to(device)
    y_train_torch = torch.FloatTensor(y_train_linear).to(device)
    
    svm_torch = SVM(lr=0.001, lambda_param=0.01, n_iters=1000)
    svm_torch.fit(X_train_torch, y_train_torch)
    predictions_torch = svm_torch.predict(X_test_torch)
    predictions_torch = np.where(predictions_torch <= 0, 0, 1)
    
    accuracy_torch = accuracy_score(y_test_linear, predictions_torch)
    print(f"SVM Accuracy: {accuracy_torch:.4f}")
