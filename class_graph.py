import collections
from scipy import sparse

class Graph:
    def __init__(self):
        self.graph = collections.defaultdict(dict)
        self.counter = 0

    def add_edge(self, u, v, weight = 1, directed = False):
        self.graph[u][v] = weight
        if not directed:
            self.graph[v][u] = weight
        self.counter+=1    

    def __str__(self):
        to_return = ''
        for vertex in self.graph:
            to_return += str(vertex) + ': '
            for edge in self.graph[vertex]:
                to_return +=  '(' + str(edge) + ', ' + str(self.graph[vertex][edge]) + ')'
                to_return += '   '

            to_return += '\n'
        return to_return

def adj_sparse(j, adj):
    n = j+1
    adj = adj.graph
    g= sparse.dok_matrix((n,n))
    for num in list(adj):
        for i in adj[num]:
            g[num, i] = adj[num].get(i)
    return g