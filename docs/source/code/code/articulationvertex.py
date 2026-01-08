"""
Given an undirected graph represented as an adjacency list, find all articulation points (cut vertices). A vertex is an articulation point if removing it increases the number of connected components of the graph.

Input Format:

    n (integer): The number of vertices in the graph (0-indexed).
    edges (list of tuples): A list of pairs (u, v) representing undirected edges between vertices u and v.

Output Format:

    A list of integers representing the articulation points in the graph. The list can be in any order.
"""

from typing import List
from typing import Set
from typing import Tuple

class Solution:
    def construct(self, n:int, edges:List[Tuple[int,int]]) -> List[List[int]]:
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        return graph

    def dfs(self, parent:int, u:int, res:Set[int]) -> None:
        self.discovered[u] = True
        self.entry[u] = self.time
        self.time += 1
        
        for v in self.graph[u]:
            if not self.discovered[v]: # tree edge
                self.treedeg[u] += 1
                self.treedeg[v] += 1
                self.dfs(u, v, res)
            elif v != parent: # back edge
                # update uplink if required
                if self.entry[v] < self.entry[self.uplink[u]]:
                    self.uplink[u] = v

        # update parent uplink if required
        if parent != -1 and self.entry[self.uplink[u]] < self.entry[self.uplink[parent]]:
            self.uplink[parent] = self.uplink[u]

        # add articulation vertices
        # case 1: root cut node
        if parent == -1 and self.treedeg[u] > 1:
            res.add(u)
        # case 2: vertex cut node
        elif self.uplink[u] == u:
            if self.treedeg[u] > 1:
                res.add(u)
            if parent != -1 and self.treedeg[parent] > 1:
                res.add(parent)
        # case 3: parent cut node (can be duplicate)
        elif self.uplink[u] == parent and self.treedeg[parent] > 1:
            res.add(parent)
        
    def findArticulationVertices(self, n:int, edges:List[Tuple[int,int]]) -> List[int]:
        self.graph = self.construct(n, edges)
        self.discovered = [False] * n
        self.uplink = list(range(n))
        self.entry = [float('inf')] * n
        self.treedeg = [0] * n
        res = set([])
        for i in range(n):
            if not self.discovered[i]:
                self.time = 0
                self.dfs(-1, i, res)
        return list(res)

if __name__ == '__main__':
    solution = Solution()
    n = 5  
    edges = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4)]
    res = solution.findArticulationVertices(n, edges)
    print(res) #[1, 3]
    n = 4  
    edges = [(0, 1), (2, 3)]
    res = solution.findArticulationVertices(n, edges)
    print(res) #[]
    n = 3  
    edges = [(0, 1), (1, 2), (2, 0)]
    res = solution.findArticulationVertices(n, edges)
    print(res) #[]
    n = 6
    edges = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 5), (5, 3)]
    res = solution.findArticulationVertices(n, edges)
    print(res) #[1, 3]
    n = 7
    edges = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 5), (5, 6), (6, 3)]
    res = solution.findArticulationVertices(n, edges)
    print(res) #[1, 3]
    n = 7
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 3)]
    res = solution.findArticulationVertices(n, edges)
    print(res) #[3]
    n = 4
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    res = solution.findArticulationVertices(n, edges)
    print(res) #[]
    n = 6
    edges = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 5)]
    res = solution.findArticulationVertices(n, edges)
    print(res) #[1, 3, 4]