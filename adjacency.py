import matplotlib.pyplot as plt
from networkx import nx
import numpy as np
import scipy.linalg  

'''  Module for functions generating adjacency matrices '''

def ferromagneticRing(N):
    ''' Ferromagnetic 1D ring
        N: elemnts in the ring
    '''

    # Diagonal bend
    J = scipy.linalg.toeplitz([0, 1] + [0] * (N - 2))

    # Closing the ring
    J[0][N - 1] = 1
    J[N - 1][0] = 1

    return J

def antiferromagneticRing(N):
    ''' Anti-ferromagnetic 1D ring 

    '''
    J = scipy.linalg.toeplitz([0, -1] + [0] * (N - 2))
    J[0][N - 1] = -1
    J[N - 1][0] = -1

    return J

