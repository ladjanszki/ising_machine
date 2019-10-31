import matplotlib.pyplot as plt
from networkx import nx
import numpy as np
import scipy.linalg
 
class HeathBath:
    ''' This class implemnts a heath bath which the ising spin can be submerged
        It can work as a termostate for examining fluctuations
        It can work as an annealer wher the temperature is set to decreasing
    '''

    def __init__(self, maxSteps, N):

        self.adjancency = None
        self.T = None
        self.maxSteps = maxSteps
        self.energy = np.empty(maxSteps)
        self.magnetization = np.empty(maxSteps)

        self.initState = 2 * np.random.randint(2, size=N) - np.ones(N, dtype=int)


    def randomNeighbour(self, state):
        ''' This function gets a binary string and flips one random bit in it
            Returning a random neighbouring string
        '''
        neighbour = np.copy(state)
        length = neighbour.shape[0]
        index = np.random.randint(length)
        neighbour[index] = neighbour[index] * -1
        return neighbour

    
    def accProb(self, oldE, newE, T):
        if newE < oldE:
            return 1.0
        else:
            return np.exp(- (newE - oldE) / T)

    
    def thermalize(self, nSteps):
        ''' Function for the first few termalisation steps '''
        pass
    

    def simulate(self, T, nSteps):
        '''
        A function to simulate a spin system for nSteps steps on T temperature
        It returns the energy and magnetization history
        '''

        # Arrays for result
        energy = np.empty(nSteps)
        magnetization = np.empty(nSteps)
         
        # Random initial state
        #np.random.seed(42)
        #initState = 2 * np.random.randint(2, size=problemSize) - np.ones(problemSize, dtype=int)
        #print(initState)
         
        # Simulated annealing optimization

        oldState = self.initState
        for i in range(self.maxSteps):
        
            #print("step: ",i)
          
            newState = self.randomNeighbour(oldState)
            #print(newState)
            #print(oldState)
          
            # If anneal decrease the temperature
            # TODO: Put switchable annealing feature back
            #if anneal:
            #    T = maxSteps / (i + 1)
            #print("Temp: ",T)

            
            T = self.maxSteps / (i + 1)
          
            # TODO: Add the bias part
            oldE = - oldState @ self.adjacency @ oldState
            newE = - newState @ self.adjacency @ newState
          
            #Save energy (per spin)
            # TODO: Normalization?
            energy[i] = oldE #/ problemSize
            
            #Save magnetization (per spin)
            # TODO: Normalization?
            self.magnetization[i] = np.sum(oldState) #/ problemSize
          
            #print("old", oldState, oldE)
            #print("new", newState, newE)
          
            ap = self.accProb(oldE, newE, T)
            #print("Acceptance probability: ", ap)
            trh =  np.random.uniform()
            #print("Trh: ", trh)
          
            if ap > trh:
                oldState = newState
                #Overwrite energy and magnetization (per spin)
                # TODO: Normalization?
                self.energy[i] = newE #/ problemSize
                self.magnetization[i] = np.sum(newState) #/ problemSize
                #print("accepted!")
          
            #print("---------------------------------")
        
        #print("Solution: ", oldState)
        
        return self.energy, self.magnetization 
   
