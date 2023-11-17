import numpy as np
import cvxpy as cp

class CvxModel:
    def __init__(self, kernel: str=None, nu: float=None, gamma: float=None, degree: float=None) -> None:
        self.kernel = kernel
        self.nu = nu 
        self.gamma = gamma 
        self.degree = degree 
        self.alphas = None
        self.rho = None
        self.kernel_function = {
            'rbf': lambda x, y: np.exp(-gamma * np.sum((x - y) ** 2)),
            'linear': lambda x, y: np.dot(x, y)
        }[kernel]
        self.X = None
        
    def getKernelMatrix(self, X):
        K = None
        if self.kernel == 'rbf':
            X_norm = np.sum(X ** 2, axis = -1)
            K = np.exp(-self.gamma * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(X, X.T)))
        elif self.kernel == 'linear':
            K = np.dot(X, X.T) + 10 ** -8 * np.eye(X.shape[0])
            
        return K
    
    def fit(self, X):
        self.X = X
        n = len(X)
        K = self.getKernelMatrix(X)
        
        alpha = cp.Variable(n)
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(K)))
        constraints = [
            alpha >= 0,  
            1 / (self.nu * n) >= alpha,
            cp.sum(alpha) == 1 
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self.alphas = alpha.value
        self.alphas[self.alphas < 10 ** -10] = 0
        
        i = self.alphas[np.where((self.alphas < 1 / (self.nu * n)) & (self.alphas > 0))].argmax()
        self.rho = sum([alpha * K[j][i] for (j, alpha) in enumerate(self.alphas)])
        
    def decision_function(self, x):
        f = sum([alpha * self.kernel_function(self.X[j], x) for (j, alpha) in enumerate(self.alphas)])
        f -= self.rho
        return int(np.sign(f))
    
    def predict(self, x):
        predictions = [0 if self.decision_function(y) == 1 else 1 for y in x]
        
        return np.array(predictions)
        
    
        