"""
Problem:

Given an undirected graph with n nodes and m edges, find all the bridges in the graph.
Input:

    n: the number of nodes (1 ≤ n ≤ 10^5)
    m: the number of edges (1 ≤ m ≤ 2 * 10^5)
    List of edges, where each edge is represented by a pair of integers (u, v) indicating an undirected edge between node u and node v.

Output:

    List of bridges represented as pairs (u, v). Each bridge should be printed in the form u v, where u < v.
"""
from typing import List

class Solution:
    def construct(self, n, edges):
        self.graph = [[] for _ in range(n)]
        for u, v in edges:
            self.graph[u].append(v)
            self.graph[v].append(u)

    def dfs(self, parent, u, res):
        self.discovered[u] = True
        self.entry[u] = self.time
        self.time += 1
        for v in self.graph[u]:
            if not self.discovered[v]: # tree edge
                self.dfs(u, v, res)
                if self.uplink[v] == v:
                    res.append([u, v])
            elif v != parent: # back edge
                if self.entry[v] < self.entry[self.uplink[u]]:
                    self.uplink[u] = v
        if parent != -1 and self.entry[self.uplink[parent]] > self.entry[self.uplink[u]]:
            self.uplink[parent] = self.uplink[u]
        
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        self.construct(n, connections)
        self.time = 0
        self.discovered = [False] * n
        self.entry = [float('inf')] * n
        self.uplink = list(range(n))
        res = []
        for i in range(n):
            if not self.discovered[i]:
                self.dfs(i, -1, res)
        return res

if __name__ == '__main__':
    solution = Solution()
    n = 5
    edges = [(1, 2),(2, 3),(3, 4),(4, 5),(2, 4)]
    res = solution.criticalConnections(n+1, edges)
    print(res)