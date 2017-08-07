import numpy as np
import plotData
from scipy.optimize import fmin_bfgs
import sigmoid

def predict(theta, X):
    
    m, n = X.shape
    p = np.zeros(m)
    
    h = sigmoid.sigmoid(X.dot(theta.T))
    
    for item in range(0, h.size):
        if h[item] > 0.5:
            p[item] = 1    
        else:
            p[item] = 0
    return p
