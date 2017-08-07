import numpy as np
import mapFeature
from pylab import contour, title, show, legend, xlabel, ylabel, plot

def plotDecisionBoundary(theta, X, y):

    if X.shape[0] == 100:
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 2]) + 2])
        plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
        plot(plot_x, plot_y)
        #legend(['Decision Boundary', 'Not admitted', 'Admitted'])
        show()
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros(shape=(len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = (mapFeature.map_feature(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))

        z = z.T
        contour(u, v, z, 1, colors='green')
        title('lambda = 1.0')
        xlabel('Microchip Test 1')
        ylabel('Microchip Test 2')
        legend(['y = 1', 'y = 0', 'Decision boundary'])
        show()