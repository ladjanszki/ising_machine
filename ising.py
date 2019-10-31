import matplotlib.pyplot as plt
from networkx import nx
import numpy as np
import scipy.linalg
from heath_bath import HeathBath
from adjacency import ferromagneticRing
from adjacency import antiferromagneticRing


# Number of spins
nSpin = 10

# Optimization steps
nSteps = 1000

# Creating the adjacency matrix
J = antiferromagneticRing(nSpin)

#print(J)
#print(type(J))

# Simulated annealing
bath = HeathBath(nSteps, nSpin)

# TODO: Setting this would be better in the constructor
bath.adjacency = J

energy, magnetization = bath.simulate()

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(energy, label='Energy per spin')
#ax.plot(magnetization, label='Average magnetization')

# Generating and plotting the solution vector
solution = np.array([2*((i+1) % 2) - 1 for i in range(nSpin)])
solE = - solution @ J @ solution
solLine = np.full(nSteps, solE)
ax.plot(solLine, label='Minimal energy')

plt.title('Ising model')
ax.legend()
plt.show()
 


# Draw the graph from adjacency matrix
#G = nx.from_numpy_matrix(J)
#
#nx.draw(G, with_labels = True)
#plt.show()


