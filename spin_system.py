import matplotlib.pyplot as plt
from networkx import nx
import numpy as np
import scipy.linalg 

''' A class for representing a spin system which can be 
    - immersed in a heath bath
    - annealed
'''

class SpinSystem:

    def __init__(nSpin, adjacency)
        # In the constructor a non interactind adj. mx. can be used (full zero)
        self.nSpin = nSpin
        self.adjacency = adjacency

        # Assigning a random starting state
        self.state = 2 * np.random.randint(2, size=N) - np.ones(N, dtype=int)


