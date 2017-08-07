from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

def plotData(X, y):

    if X.shape[0] == 100:
        
        pos = where(y == 1)
        neg = where(y == 0)
        scatter(X[neg, 0], X[neg, 1], marker='+', c='b')
        scatter(X[pos, 0], X[pos, 1], marker='o', c='y')
        xlabel('Exam 1 score')
        ylabel('Exam 2 score')
        legend(['Not Admitted', 'Admitted'])
        #show()
    else:
        pos = where(y == 1)
        neg = where(y == 0)
        scatter(X[pos, 0], X[pos, 1], marker='+', c='b')
        scatter(X[neg, 0], X[neg, 1], marker='o', c='y')
        #show()