import numpy as np

def sigmoid(z):
        
    g = float(1.0) / (1.0 +  np.exp( -1.0 * z ))

    return g