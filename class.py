import collections

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