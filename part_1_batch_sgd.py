import numpy as np
import plotData
from scipy.optimize import fmin_bfgs
import sigmoid

def part_1_batch_sgd(X, y, theta, learning_rate, max_iters, tolerance):
    
    cost = 1
    iter = 0
    
    while (cost > tolerance and iter < max_iters):
        
        J = costFunction(theta, X, y)      
        print (J)
        
        gradient = costGradient(theta, X, y)
        #print (gradient)
        #print (gradient.shape)
        
        theta = theta - learning_rate * gradient
        
        iter = iter + 1
        
    return theta

    
def costFunction(theta, X, y):
  
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.T))
           
    J = (1./m) * ( (-y.T.dot(np.log(h))) - (1.0-y).T.dot(np.log(1.0-h)) )
    
    return J

def costGradient(theta, X, y):
    
    h = sigmoid.sigmoid(X.dot(theta.T))   
    error = h - y   
    
    grad = X.T.dot(error) / y.size
    #print (grad)

    return grad
        
        