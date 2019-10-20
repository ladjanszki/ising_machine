import matplotlib.pyplot as plt
import networkx as nx

G=nx.Graph()
G.add_node(1,pos=(1,1))
G.add_node(2,pos=(2,2))
G.add_edge(1,2)

pos=nx.get_node_attributes(G,'pos')

nx.draw(G, pos, with_labels=True)
plt.axis("on")
plt.show()
