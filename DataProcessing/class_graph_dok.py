import numpy as np
from scipy.sparse import dok_matrix

class Graph:
    def __init__(self, node):
        self.graph = dok_matrix((node,node), dtype = np.float32)
        self.counter = 0

    def add_edge(self, u, v, weight = 1, directed = False):
        self.graph[u,v] = weight
        if not directed:
            self.graph[v,u] = weight
        self.counter+=1  

    def to_csr(self):
        return self.graph.tocsr()

    def __str__(self):
        return str(self.graph)