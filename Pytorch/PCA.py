import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.total_variance = None
        
    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean
        
        cov_matrix = torch.mm(X_centered.T, X_centered) / (X.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        self.components = sorted_eigenvectors[:, :self.n_components]
        self.eigenvalues = sorted_eigenvalues[:self.n_components]
        self.total_variance = torch.sum(torch.var(X, dim=0))
        
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        X_centered = X - self.mean
        return torch.mm(X_centered, self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        return torch.mm(X_transformed, self.components.T) + self.mean
    
    def explained_variance_ratio(self):
        return self.eigenvalues / self.total_variance

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== PCA on Classification Dataset ===")
    X_cls, y_cls = datasets.make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                               n_redundant=3, random_state=42)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=1234)
    
    pca_torch = PCA(n_components=2)
    X_torch = torch.FloatTensor(X_train_cls).to(device)
    X_transformed_torch = pca_torch.fit_transform(X_torch)
    
    explained_var_torch = pca_torch.explained_variance_ratio()
    print(f"PyTorch explained variance ratio: {explained_var_torch.cpu().numpy()}")
