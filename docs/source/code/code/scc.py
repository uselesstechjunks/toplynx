"""
Find Strongly Connected Components in a Directed Graph

Implement a function to find all SCCs in a directed graph using Kosaraju’s Algorithm. Your function should take as input the number of nodes and a list of directed edges and output the SCCs as a list of lists, where each list contains the nodes in one SCC.
"""
class Solution:
    def construct(self, n, edges, transpose=False):
        graph = [[] for _ in range(n)]
        for u, v in edges:
            if not transpose:
                graph[u].append(v)
            else:
                graph[v].append(u)
        return graph
        
    def dfs(self, graph, parent, u, curr):
        self.discovered[u] = True
        for v in graph[u]:
            if not self.discovered[v]: # tree edge
                self.dfs(graph, u, v, curr)
        curr.append(u)

    def scc(self, n, edges):
        g = self.construct(n, edges, transpose=False)
        gt = self.construct(n, edges, transpose=True)
        
        self.discovered = [False] * n
        order = []
        for u in range(n):
            if not self.discovered[u]:
                self.dfs(g, -1, u, order)
        self.discovered = [False] * n
        
        res = []
        roots = [-1] * n
        for u in reversed(order):            
            if not self.discovered[u]:
                curr = []
                self.dfs(gt, -1, u, curr)
                res.append(curr)
                root = min(curr)
                for v in curr:
                    roots[v] = root
        print(roots)

        condensed = [[] for _ in range(n)]
        for u in range(n):
            for v in g[u]:
                if roots[u] != roots[v]:
                    condensed[roots[u]].append(roots[v])
        return res, condensed

def test_scc():
    solution = Solution()
    # Test Case 1
    n = 5
    edges = [(1, 0), (0, 2), (2, 1), (0, 3), (3, 4)]
    res, _ = solution.scc(n, edges)
    print(res) # [[4], [3], [0, 2, 1]]
    # Test Case 2 (Edge Case: No edges)
    n = 3
    edges = []
    res, _ = solution.scc(n, edges)
    print(res) # [[0], [1], [2]]
    # Test Case 3 (Single SCC)
    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    res, _ = solution.scc(n, edges)
    print(res) # [[0, 1, 2, 3]]
    # Test Case 4 (Disconnected graph)
    n = 6
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    res, _ = solution.scc(n, edges)
    print(res) # [[0, 2, 1], [3, 5, 4]]

def test_cond():
    solution = Solution()
    # Test Case 1
    n = 6
    dependencies = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (2, 3)]
    res, condensed = solution.scc(n, dependencies)
    print(res) # SCCs: [[0, 2, 1], [3, 5, 4]]  
    print(condensed)

if __name__ == '__main__':
    #test_scc()
    test_cond()