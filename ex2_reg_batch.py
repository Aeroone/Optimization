import numpy as np
import plotData
from scipy.optimize import fmin_bfgs
import pylab as pl
import costFunction_reg
import plotDecisionBoundary
import sigmoid
import predict
import mapFeature
import batch_sgd

if __name__=="__main__":
    
    #load the dataset
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = data[:, 0:2]
    #print (X.shape)
    y = data[:, 2]
    m, n = X.shape
  
    new_data = mapFeature.map_feature(X[:,0],X[:,1])
        
    for i in range(2,new_data.shape[1]):
        m = np.mean(new_data[:, i])
        s = np.std(new_data[:, i])
        new_data[:, i] = (new_data[:, i] - m) / s
        
    theta = 0.1* np.zeros(new_data.shape[1])
        
    array = batch_sgd.batch_sgd(new_data, y, theta, 0.01, 400, -5, 10)
    #print (array)
    lr1 = array[:, 28].tolist()
    
    array = batch_sgd.batch_sgd(new_data, y, theta, 0.03, 400, -5, 10)
    #print (array)
    lr2 = array[:, 28].tolist()
    
    array = batch_sgd.batch_sgd(new_data, y, theta, 0.05, 400, -5, 10)
    #print (array)
    lr3 = array[:, 28].tolist()
    
    array = batch_sgd.batch_sgd(new_data, y, theta, 0.1, 400, -5, 10)
    #print (array)
    lr4 = array[:, 28].tolist()
    
    array = batch_sgd.batch_sgd(new_data, y, theta, 0.3, 400, -5, 10)
    #print (array)
    lr5 = array[:, 28].tolist()
    
    array = batch_sgd.batch_sgd(new_data, y, theta, 0.5, 400, -5, 10)
    #print (array)
    lr6 = array[:, 28].tolist()
    
    pl.figure(1)
    pl.plot(lr1)
    pl.title('cost vs. number of iterations (learning rate:0.01)')
    pl.xlabel('number of iterations')
    pl.ylabel('cost')
    
    pl.figure(2)
    pl.plot(lr2)
    pl.title('cost vs. number of iterations (learning rate:0.03)')
    pl.xlabel('number of iterations')
    pl.ylabel('cost')
    
    pl.figure(3)
    pl.plot(lr3)
    pl.title('cost vs. number of iterations (learning rate:0.05)')
    pl.xlabel('number of iterations')
    pl.ylabel('cost')
    
    pl.figure(4)
    pl.plot(lr4)
    pl.title('cost vs. number of iterations (learning rate:0.1)')
    pl.xlabel('number of iterations')
    pl.ylabel('cost')
    
    pl.figure(5)
    pl.plot(lr5)
    pl.title('cost vs. number of iterations (learning rate:0.3)')
    pl.xlabel('number of iterations')
    pl.ylabel('cost')
    
    pl.figure(6)
    pl.plot(lr6)
    pl.title('cost vs. number of iterations (learning rate:0.5)')
    pl.xlabel('number of iterations')
    pl.ylabel('cost')
    pl.show()# show the plot on the screen
    