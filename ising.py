import matplotlib.pyplot as plt
from networkx import nx
import numpy as np
import scipy.linalg

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


problemSize = 20
maxSteps = 2000

energy = np.empty(maxSteps)




# Random initial guess
#np.random.seed(42)
initState = 2 * np.random.randint(2, size=problemSize) - np.ones(problemSize, dtype=int)
#print(initState)

# Generating ANTIFERROMAGNETIC  adjacency matrix
J = scipy.linalg.toeplitz([0, -1] + [0] * (problemSize - 2))

# Adjacency matrix for 1D Antiferromagnetic RING
# Adding the ring connections
J[0][problemSize - 1] = -1
J[problemSize - 1][0] = -1

#print(J)

# Generating the solution vector
solution = np.array([2*((i+1) % 2) - 1 for i in range(problemSize)])
#print(solution)
solE = - solution @ J @ solution
solLine = np.full(maxSteps, solE)

# Optimization loop
oldState = initState
for i in range(maxSteps):

  print("step: ",i)

  newState = randomNeighbour(oldState)
  #print(newState)
  #print(oldState)

  T = maxSteps / (i + 1)

  print("Temp: ",T)


  # TODO: Add the bias part
  oldE = - oldState @ J @ oldState
  newE = - newState @ J @ newState

  #Save energy
  energy[i] = oldE

  print("old", oldState, oldE)
  print("new", newState, newE)

  ap = accProb(oldE, newE, T)
  print("Acceptance probability: ", ap)
  trh =  np.random.uniform()
  print("Trh: ", trh)

  if ap > trh:
    oldState = newState
    #Overwrite energy is new state accepted
    energy[i] = newE
    print("accepted!")

  print("---------------------------------")

print("Solution: ", oldState)

plt.plot(energy)
plt.plot(solLine)
plt.show()




# Draw the graph from adjacency matrix
#G = nx.from_numpy_matrix(J)
#
#nx.draw(G, with_labels = True)
#plt.show()


