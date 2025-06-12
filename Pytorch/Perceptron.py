import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.criterion = nn.BCEWithLogitsLoss()
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = torch.zeros(n_features, 1, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)
        
        for i in range(self.n_iters):
            linear_output = torch.mm(X, self.weights) + self.bias
            loss = self.criterion(linear_output, y)
            loss.backward()
            
            with torch.no_grad():
                self.weights -= self.lr * self.weights.grad
                self.bias -= self.lr * self.bias.grad
                self.weights.grad.zero_()
                self.bias.grad.zero_()
    
    def predict(self, X):
        with torch.no_grad():
            linear_output = torch.mm(X, self.weights) + self.bias
            predictions = torch.sigmoid(linear_output)
            return (predictions > 0.5).float()
    
    def predict_proba(self, X):
        with torch.no_grad():
            linear_output = torch.mm(X, self.weights) + self.bias
            return torch.sigmoid(linear_output)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X, y = datasets.make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, 
                                       random_state=1, n_clusters_per_class=1, class_sep=2.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    print("=== Modern Perceptron (with sigmoid and BCE loss) ===")
    model_modern = Perceptron(lr=0.01, n_iters=1000)
    model_modern.fit(X_train, y_train)
    predictions_modern = model_modern.predict(X_test)
    probabilities_modern = model_modern.predict_proba(X_test)
    
    predictions_modern_np = predictions_modern.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    with torch.no_grad():
        linear_output = torch.mm(X_test, model_modern.weights) + model_modern.bias
        final_loss = model_modern.criterion(linear_output, y_test).item()
    
    print(f"Final BCE Loss: {final_loss:.4f}")
    
    accuracy_modern = accuracy_score(y_test_np, predictions_modern_np)
    print(f"Accuracy: {accuracy_modern:.4f}")
    
    with torch.no_grad():
        weights_modern = model_modern.weights.cpu().numpy().flatten()
        bias_modern = model_modern.bias.item()
        print(f"Learned weights: {weights_modern}")
        print(f"Learned bias: {bias_modern:.4f}")
    
    
    plt.figure(figsize=(15, 5))
    X_test_np = X_test.cpu().numpy()
    
    colors_modern = ['red' if pred == 0 else 'blue' for pred in predictions_modern_np.flatten()]
    plt.scatter(X_test_np[:, 0], X_test_np[:, 1], c=colors_modern, alpha=0.7)
    plt.title(f'Modern Perceptron\nAccuracy: {accuracy_modern:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    

    plt.show()
