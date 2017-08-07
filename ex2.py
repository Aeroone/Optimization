import numpy as np
import plotData
from scipy.optimize import fmin_ncg
import costFunction
import plotDecisionBoundary
import sigmoid
import predict
import part_1_batch_sgd

def optimization(added_X, y):
    def f(theta):
        return costFunction.costFunction(theta, added_X, y)

    def fprime(theta):
        return costFunction.costGradient(theta, added_X, y)

    #Initialize theta parameters
    theta = 0.1* np.random.randn(3)

    return fmin_ncg(f, theta, fprime, disp=True, maxiter=500)   


if __name__=="__main__":
    
    #load the dataset
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = data[:, 0:2]
    #print (X.shape)
        
    y = data[:, 2]
        
    m, n = X.shape
    
    plotData.plotData(X, y)
    
    added_X = np.ones(shape=(m, 3))
    added_X[:, 1:3] = X
    #print (added_X.shape)
                
    theta = np.zeros(3)
    #theta = optimization(added_X, y)
        
    theta = part_1_batch_sgd.part_1_batch_sgd(added_X, y, theta, 0.001, 100000, 0)
    #print (theta)
       
    test_data = np.array([1.0, 45.0, 85.0])
    #probability = sigmoid.sigmoid((test_data).dot(theta.T))
    
    #print ('For a student with scores 45 and 85, we predict and admission ' + \
    #'probability of %f' %probability)
    
    p = predict.predict(theta, added_X)    
    print (p)
    
    accuracy = ((y[np.where(p == y)].size / float(y.size)) * 100.0)
    print ('Train Accuracy: %f' % accuracy)
    
    plotDecisionBoundary.plotDecisionBoundary(theta, added_X, y)
    