import numpy as np
import sigmoid

def costFunction(theta, X, y):

    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.T)) # predicted probability of label 1
    
    #if 1 in h.tolist():
    #    print('find 1 in h!')
    #    h_list = h.tolist()
    #    for item in h_list:
    #        if item == 1.0:
    #            item = item + np.exp( -1.0 * 20 ) 
    #    h = np.array(h_list)
    
    J = (1./m) * ( (-y.T.dot(np.log(h))) - (1.0-y).T.dot(np.log(1.0-h)) )
    print (J)
    
    return J

def costGradient(theta, X, y):
          
    h = sigmoid.sigmoid(X.dot(theta.T))   
    error = h - y   
    #print (error.shape)
    
    grad = X.T.dot(error.T) / y.size
    #print (X.T.shape)
    #print (error.T.shape)
    
    #print (grad)
    #print (grad.shape)
    
    return grad
    