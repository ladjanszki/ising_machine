import matplotlib.pyplot as plt
from networkx import nx
import numpy as np
import scipy.linalg

# TODO: Generate the random numbers all at one in the beginning
# TODO: Warm-up the random generator.

def randomNeighbour(state):
  neighbour = np.copy(state)
  length = neighbour.shape[0]
  index = np.random.randint(length)
  neighbour[index] = neighbour[index] * -1
  return neighbour

def accProb(oldE, newE, T):
  if newE < oldE:
    return 1.0
  else:
    return np.exp(- (newE - oldE) / T)

def isingOpt(maxSteps, problemSize, T, J):
  # Arrays for result
  energy = np.empty(maxSteps)
  magnetization = np.empty(maxSteps)
   
  # Random initial state
  #np.random.seed(42)
  initState = 2 * np.random.randint(2, size=problemSize) - np.ones(problemSize, dtype=int)
  #print(initState)
   
  # Simulated annealing optimization
  oldState = initState
  for i in range(maxSteps):
  
    #print("step: ",i)
  
    newState = randomNeighbour(oldState)
    #print(newState)
    #print(oldState)
  
    # If anneal decrease the temperature
    if anneal:
      T = maxSteps / (i + 1)
    #print("Temp: ",T)
  
  
    # TODO: Add the bias part
    oldE = - oldState @ J @ oldState
    newE = - newState @ J @ newState
  
    #Save energy (per spin)
    energy[i] = oldE / problemSize
    
    #Save magnetization (per spin)
    magnetization[i] = np.sum(oldState) / problemSize
  
    #print("old", oldState, oldE)
    #print("new", newState, newE)
  
    ap = accProb(oldE, newE, T)
    #print("Acceptance probability: ", ap)
    trh =  np.random.uniform()
    #print("Trh: ", trh)
  
    if ap > trh:
      oldState = newState
      #Overwrite energy and magnetization (per spin)
      energy[i] = newE / problemSize
      magnetization[i] = np.sum(newState) / problemSize
      #print("accepted!")
  
    #print("---------------------------------")
  
  #print("Solution: ", oldState)
  
  #fig = plt.figure()
  #ax = plt.subplot(111)
  #ax.plot(energy, label='Energy per spin')
  #ax.plot(magnetization, label='Average magnetization')
  #plt.title('Ising model')
  #ax.legend()
  #plt.show()

  return np.mean(energy), np.mean(magnetization) 

 
# TODO: Think of scaling of the teperature path when simulated annealing
T = 10 # Initial temperature
anneal = False

# Number of spins
problemSize = 10

# Number of opt. steps
maxSteps = 2000
 





# Ferromagnetic 1D ring adjacencey matrix
J = scipy.linalg.toeplitz([0, 1] + [0] * (problemSize - 2))
J[0][problemSize - 1] = 1
J[problemSize - 1][0] = 1

## Anti-ferromagnetic 1D ring adjacencey matrix
#J = scipy.linalg.toeplitz([0, -1] + [0] * (problemSize - 2))
#J[0][problemSize - 1] = -1
#J[problemSize - 1][0] = -1
#print(J)

## Generating the solution vector
#solution = np.array([2*((i+1) % 2) - 1 for i in range(problemSize)])
##print(solution)
#solE = - solution @ J @ solution
#solLine = np.full(maxSteps, solE)



meanE, meanM = isingOpt(maxSteps, problemSize, T, J)

print(T)
print(meanE)
print(meanM)




# Draw the graph from adjacency matrix
#G = nx.from_numpy_matrix(J)
#
#nx.draw(G, with_labels = True)
#plt.show()


