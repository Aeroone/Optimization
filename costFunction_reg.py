import numpy as np
import sigmoid

def costFunction(theta, X, y, lamda):

    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.T))
    thetaR = theta[1:]
    
    #thetaR = theta[1:, 0]    
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
    