import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

class CvxModel:
    def __init__(self, kernel: str=None, nu: float=None, gamma: float=None, degree: float=None) -> None:
        self.kernel = kernel
        self.nu = nu 
        self.gamma = gamma 
        self.degree = degree 
        self.alphas = None
        self.rho = None
        self.K = None
        self.kernel_function = {
            'rbf': lambda x, y: rbf_kernel(x, y, gamma=self.gamma),
            'linear': lambda x, y: linear_kernel(x, y)
        }[kernel]
        self.X = None
        
    def getKernelMatrix(self, X):
        K = self.kernel_function(X, X)
        
        return K
    
    def compute_alphas(self):
        n = len(self.X)
        K = self.K 
        alpha = cp.Variable(n)
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(K)))
        constraints = [
            alpha >= 0,  
            1 >= alpha,
            cp.sum(alpha) == self.nu * n
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return alpha.value
    
    def compute_rho(self):
        n = len(self.X)
        i = self.alphas[np.where((self.alphas < 1) & (self.alphas > 0))].argmax()
        
        return np.sum(self.alphas * self.K[ : , i])
    
    def fit(self, X):
        self.X = X
        self.K = self.getKernelMatrix(X)
        self.alphas = self.compute_alphas()
        self.rho = self.compute_rho()
        
    def decision_function(self, Y):
        f = np.sum(self.alphas[ : , np.newaxis] * self.kernel_function(self.X, Y), axis=0)
        f -= self.rho
        
        return np.sign(f).astype(int)
    
    def predict(self, Y):
        predictions = self.decision_function(Y)
        predictions[predictions == 1] = 0
        predictions[predictions == -1] = 1
        
        return np.array(predictions)
        
    
        