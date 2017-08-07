import numpy as np
import plotData
from scipy.optimize import fmin_bfgs
import costFunction_reg
import plotDecisionBoundary
import sigmoid
import predict
import mapFeature

def optimization(X, y, lamda):
    def f(theta):
        return costFunction_reg.costFunction(theta, X, y, lamda)

    def fprime(theta):
        return costFunction_reg.costGradient(theta, X, y, lamda)

    #Initialize theta parameters
    theta = 0.1* np.random.randn(28)

    return fmin_bfgs(f, theta, fprime, disp=True, maxiter=500) 

if __name__=="__main__":
    
    #load the dataset
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = data[:, 0:2]
    #print (X.shape)
    y = data[:, 2]
    m, n = X.shape
    
    plotData.plotData(X, y)
    #print (X.shape)
    new_data = mapFeature.map_feature(X[:,0],X[:,1])
    #print (new_data)
    #print (new_data.shape)
    
    #theta = 0.1* np.zeros(new_data.shape[1])
    theta = 0.1* np.random.randn(new_data.shape[1])
    #print (theta)
    #print (theta.shape)
    
    #set lamda
    lamda = 1.0
    
    J = costFunction_reg.costFunction(theta, new_data, y, lamda)
    #print (J)
    
    #costFunction_reg.costGradient(theta, new_data, y, lamda)
    theta = optimization(new_data, y, lamda)
    print (theta)
    
    p = predict.predict(theta, new_data)    
    print (p)
    
    accuracy = ((y[np.where(p == y)].size / float(y.size)) * 100.0)
    print ('Train Accuracy: %f' % accuracy)    
    
    plotDecisionBoundary.plotDecisionBoundary(theta, X, y)      