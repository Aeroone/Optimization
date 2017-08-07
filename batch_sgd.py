import numpy as np
import plotData
from scipy.optimize import fmin_bfgs
import sigmoid
import math

def batch_sgd(X, y, theta, learning_rate, max_iters, tolerance, batch_size):
    
    mini_batch_size = batch_size
    cost = 1
    J_old = 0
    iter = 0
    lamda = 0.1
    diff = math.pow(10, tolerance)
    
    array = []   
    while (cost > diff and iter < max_iters):
        
        ###select mini-batch###
        low_number = (mini_batch_size * iter) % 118
        up_number = mini_batch_size * (iter + 1) % 118
        #print (low_number)
        #print (up_number)
        
        if (mini_batch_size * iter) % 118 >= (118 - mini_batch_size):
            
            X_selected_1 = X[low_number:X.shape[0], :]
            X_selected_2 = X[0:up_number, :]
            X_list = X_selected_1.tolist() + X_selected_2.tolist()
            X_selected = np.array(X_list) 
        
            y_selected_1 = y[low_number:y.shape[0]]
            y_selected_2 = y[0:up_number]
            y_list = y_selected_1.tolist() + y_selected_2.tolist()
            y_selected = np.array(y_list)
        
        else:
            X_selected = X[low_number:up_number, :]    
            y_selected = y[low_number:up_number]
        #######################
        
        #print (X_selected)
        #print (iter)
                
        J = costFunction(theta, X, y, lamda)      
        gradient = costGradient(theta, X_selected, y_selected, lamda)
        print (J)
        
        if (J - J_old) > 0:
            cost = J - J_old
        else:
            cost = J_old - J
           
        J_old = J
        
        tmp_array = theta.tolist() + [J]
        
        array.append(tmp_array)
        
        theta = theta - learning_rate * gradient
                    
        iter = iter + 1
        
    return np.array(array)

    
def costFunction(theta, X, y, lamda):

    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.T))
    thetaR = theta[1:]
    
    J = (1./m) * ( (-y.T.dot(np.log(h))) - ((1.0-y).T.dot(np.log(1.0-h))) ) \
        + (lamda / (2.0 * m)) * (thetaR.T.dot(thetaR))
    
    return J

def costGradient(theta, X, y, lamda):
    
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.T))
    
    thetaR = theta[1:]
    #print (thetaR)
    #print (thetaR.shape)
    error = h - y
    sumerror = error.T.dot(X[:, 1])
    gradient0 = (1.0 / m) * sumerror
    
    XR = X[:, 1:X.shape[1]]
    sumerror = error.T.dot(XR)
    
    gradient = (1.0 / m) * (sumerror + lamda * thetaR)
    
    gradient_list = []
    gradient_list.append(gradient0)
    gradient_list = gradient_list + gradient.tolist()
    #print (gradient_list)
    gradient = np.array(gradient_list)
    #print (gradient)
    #print (gradient.shape)
    return gradient        
        