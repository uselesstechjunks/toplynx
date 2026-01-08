class Solution:
    def tsort(self, n, edges):
        def dfs(adj, discovered, processed, u, order):
            discovered[u] = True
            for v in adj[u]:
                if not discovered[v]:
                    dfs(adj, discovered, processed, v, order)
                elif not processed[v]:
                    return # back edge, not a DAG
            processed[u] = True
            order.insert(0, u)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
        discovered = [False] * n
        processed = [False] * n
        order = []
        for u in range(n):
            if not discovered[u]:
                dfs(adj, discovered, processed, u, order)
        return adj, order if len(order) == n else []

    def longest(self, n, edges):
        adj, tsorted = self.tsort(n, edges)
        d = [float('-inf')] * n # d[v]: count of vertices in the longest path from 1 to v
        d[1] = 1
        parent = [-1] * n
        for u in tsorted:
            for v in adj[u]:
                if d[v] < d[u] + 1:
                    parent[v] = u
                    d[v] = d[u] + 1
        def findPath(parent, u, path):
            if u == -1: return
            findPath(parent, parent[u], path)
            path.append(u)
        path = []
        if d[n-1] != float('-inf'):
            findPath(parent, n-1, path)
        return d[n-1], path

if __name__ == '__main__':
    solution = Solution()
    n = 6
    edges = [
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6]
    ]
    length = solution.longest(n+1, edges)
    print(length)