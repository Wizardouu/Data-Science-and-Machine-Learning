import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    return torch.mean(-y_true * torch.log(y_pred + 1e-8) - (1 - y_true) * torch.log(1 - y_pred + 1e-8)).item()

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-torch.clamp(z, -250, 250)))  # Clamp to prevent overflow
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = torch.zeros(n_features, 1, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)
        
        for i in range(self.n_iters):
            linear_model = torch.mm(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            criterion = nn.BCELoss()  # Creates the loss function
            loss = criterion(y_predicted, y)
            loss.backward()
            
            with torch.no_grad():
                self.weights -= self.lr * self.weights.grad
                self.bias -= self.lr * self.bias.grad
                self.weights.grad.zero_()
                self.bias.grad.zero_()
    
    def predict(self, X):
        with torch.no_grad():
            linear_model = torch.mm(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            predictions = (y_predicted > 0.5).float()
            return predictions
    
    def predict_proba(self, X):
        with torch.no_grad():
            linear_model = torch.mm(X, self.weights) + self.bias
            return self.sigmoid(linear_model)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = datasets.make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, 
                                       random_state=1, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    model = LogisticRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    predictions_np = predictions.cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    bce = binary_cross_entropy(y_test, probabilities)
    print(f"Binary Cross Entropy: {bce:.4f}")
    
    accuracy = accuracy_score(y_test_np, predictions_np)
    print(f"Accuracy: {accuracy:.4f}")
    
    with torch.no_grad():
        weights = model.weights.cpu().numpy().flatten()
        bias = model.bias.item()
        print(f"Learned weights: {weights}")
        print(f"Learned bias: {bias:.4f}")
    
    # Create decision boundary plot
    plt.figure(figsize=(12, 5))
    X_test_np = X_test.cpu().numpy()
    colors = ['red' if pred == 0 else 'blue' for pred in predictions_np.flatten()]
    plt.scatter(X_test_np[:, 0], X_test_np[:, 1], c=colors, alpha=0.7)
    plt.title('Predictions on Test Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
