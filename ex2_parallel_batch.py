import numpy as np
import sigmoid
import mapFeature
import multiprocessing
from multiprocessing import Pool, Value  
import math
import pylab as pl
import time

#load the dataset
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
#print (X.shape)
y = data[:, 2]
m, n = X.shape
  
new_data = mapFeature.map_feature(X[:,0],X[:,1])
       
#########data standarized#########
##################################        
for i in range(2,new_data.shape[1]):
    m = np.mean(new_data[:, i])
    s = np.std(new_data[:, i])
    new_data[:, i] = (new_data[:, i] - m) / s
##################################
##################################


############parameters############
##################################
lamda = 0.1
learning_rate = 0.1
max_iters = 400
tolerance = math.pow(10, -5)
batch_size = 10
nthreads = 4
##################################
##################################

J = []
counter = Value('d', 1)
theta = 0.1* np.zeros(new_data.shape[1])

def run(blank):  
    global counter, theta, J  
    counter.value = counter.value + 1
    tmp_J = costFunction(theta, new_data, y, lamda)
    #print (counter.value, tmp_J)
    J.append(tmp_J)
    
    X_selected, y_selected = get_batch(batch_size, new_data, y, int(counter.value))
    gradient = costGradient(theta, X_selected, y_selected, lamda) 
    theta = theta - learning_rate * gradient 
    return J
    

def parallel_batch_sgd(X, y, theta, learning_rate, nthreads, max_iters, tolerance):
    
    #print (max_iters)
    #print (nthreads)
    pool = Pool(nthreads)  
    J = pool.map(run, range(max_iters))  
    pool.close()  
    pool.join() 
    return J  

def get_batch(batch_size, X, y, iter):
    
    ###select mini-batch###
    low_number = (batch_size * iter) % 118
    up_number = batch_size * (iter + 1) % 118
    #print (low_number)
    #print (up_number)
        
    if (batch_size * iter) % 118 >= (118 - batch_size):
            
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
        
    return X_selected, y_selected     
    #######################

   
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

if __name__=="__main__":

    multiprocessing.freeze_support()
    
    cost_sum = 0
    Monte_Carlo = 1
    for i in range(Monte_Carlo):
        #print (i)
        start = time.clock()
        J = parallel_batch_sgd(new_data, y, theta, learning_rate, nthreads, max_iters, tolerance)
        end = time.clock()
        cost = end - start #time in second
        cost_sum = cost_sum + cost
    
    #print (J)
    index = 0
    for i in range(0,len(J)):
        if (len(J[i])> index):
            index = i
            
    print (J[index])
    #print (len(J[index]))
    
    cost_sum = cost_sum/Monte_Carlo
    print ('run time %f' % cost_sum)
    
    pl.figure(1)
    pl.plot(J[index])
    pl.title('cost vs. number of iterations (learning rate:%f)'% learning_rate)
    pl.xlabel('number of iterations')
    pl.ylabel('cost')
    pl.show()
 