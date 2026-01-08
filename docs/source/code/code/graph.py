from collections import deque

class Solution:
    def bfs1(self, digraph, src):
        n = len(digraph)
        queue = deque([src])
        discovered = [False] * n
        discovered[src] = True
        vertices = []
        edges = []
        while queue:
            u = queue.popleft()
            vertices.append(u)
            for v in digraph[u]:
                edges.append((u, v))
                if not discovered[v]:
                    discovered[v] = True
                    queue.append(v)
        return vertices, edges
    
    def bfs2(self, graph, src):
        n = len(graph)
        queue = deque([src])
        discovered = [False] * n
        processed = [False] * n
        discovered[src] = True
        vertices = []
        edges = []
        while queue:
            u = queue.popleft()            
            vertices.append(u)
            for v in graph[u]:
                if not processed[v]:
                    edges.append((u, v))
                if not discovered[v]:
                    discovered[v] = True
                    queue.append(v)
            processed[u] = True
        return vertices, edges

    def hasCycle(self, digraph, n):
        def dfs(digraph, discovered, processed, u):
            discovered[u] = True
            for v in digraph.get(u, []):
                if not discovered[v]:
                    if dfs(digraph, discovered, processed, v):
                        return True
                elif not processed[v]:
                    return True
            processed[u] = True
            return False
        discovered = [False] * n
        processed = [False] * n
        for u in range(n):
            if not discovered[u] and dfs(digraph, discovered, processed, u):
                return True
        return False
    
    def hasCycleUndirected(self, graph, n):
        def dfs(graph, discovered, parent, u):
            discovered[u] = True
            for v in graph.get(u, []):
                if parent == v:
                    continue
                if discovered[v]:
                    return True
                else:
                    if dfs(graph, discovered, u, v):
                        return True
            return False
        discovered = [False] * n
        for u in range(n):
            if not discovered[u] and dfs(graph, discovered, -1, u):
                return True
        return False

    def isBipartite(self, graph, n):
        colors = [None] * n
        def bfs(graph, colors, src):
            queue = deque([src])
            colors[src] = 0
            while queue:
                u = queue.popleft()
                for v in graph.get(u, []):
                    if colors[v] is None:
                        colors[v] = colors[u] ^ 1
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        return False
            return True
        for u in range(n):
            if colors[u] is None and not bfs(graph, colors, u):
                return False
        return True
    
    def findBCC(self, graph, n):        
        def dfs1(parent, u):
            nonlocal discovered, entry, low, time, articulation, bridges
            discovered[u] = True
            low[u], entry[u] = time, time
            time += 1
            treedeg = 0
            isArticulation = False
            for v in graph.get(u, []):
                if not discovered[v]:
                    treedeg += 1
                    dfs1(u, v)
                    low[u] = min(low[u], low[v])
                    if low[v] > entry[u]:
                        bridges.append((u, v))
                    if low[v] >= entry[u]:
                        isArticulation = True # parent articulation
                elif parent != v:
                    low[u] = min(low[u], entry[v])
            if (parent == -1 and treedeg > 1) or (parent != -1 and isArticulation):
                articulation.append(u)
        def dfs2(parent, u, depth):
            nonlocal discovered, entry, low, articulation, bridges
            discovered[u] = True
            low[u], entry[u] = depth, depth
            treedeg = 0
            isArticulation = False
            for v in graph.get(u, []):
                if not discovered[v]:
                    treedeg += 1
                    dfs2(u, v, depth + 1)
                    low[u] = min(low[u], low[v])
                    if low[v] > entry[u]:
                        bridges.append((u, v))
                    if low[v] >= entry[u]:
                        isArticulation = True # parent articulation
                elif parent != v:
                    low[u] = min(low[u], entry[v])
            if (parent == -1 and treedeg > 1) or (parent != -1 and isArticulation):
                articulation.append(u)
        discovered = [False] * n
        entry = [None] * n
        low = [None] * n
        articulation, bridges = [], []
        time = 0
        for u in range(n):
            if not discovered[u]:
                dfs1(-1, u)
        a, b = articulation, bridges
        discovered = [False] * n
        entry = [None] * n
        low = [None] * n
        articulation, bridges = [], []
        for u in range(n):
            if not discovered[u]:
                dfs2(-1, u, 0)
        c, d = articulation, bridges
        assert(a == c and b == d)
        return articulation, bridges
    def tsort(self, digraph, n):
        if self.hasCycle(digraph, n):
            return []
        def tarjan():
            order = []
            indeg = [0] * n
            for u in digraph:
                for v in digraph[u]:
                    indeg[v] += 1
            queue = deque([x for x in range(n) if indeg[x] == 0])
            while queue:
                u = queue.popleft()
                order.append(u)
                for v in digraph[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        queue.append(v)
            return order
        def recursive():
            stack = []
            visited = [False] * n
            def dfs(u):
                nonlocal stack, visited
                visited[u] = True
                for v in digraph[u]:
                    if not visited[v]:
                        dfs(v)
                stack.append(u)
            for u in range(n):
                if not visited[u]:
                    dfs(u)
            stack.reverse()
            return stack
        order = tarjan()
        stack = recursive()
        return stack

def digraphSearch():
    solution = Solution()
    digraph = {0: [1, 2],1: [3, 4],2: [5, 6],3: [],4: [],5: [],6: []}
    start_node = 0
    res = solution.bfs1(digraph, start_node)
    print(res)

def undirectedGraphSearch():
    solution = Solution()
    graph = {0: [1, 3],1: [0, 2],2: [1, 3],3: [0, 2]}
    start_node = 0
    res = solution.bfs2(graph, start_node)
    print(res)

def cycleDirected():
    solution = Solution()
    digraph, n = {}, 0 # Should return False
    res = solution.hasCycle(digraph, n)
    print(res)
    digraph, n = {0: []}, 1  # Should return False
    res = solution.hasCycle(digraph, n)
    print(res)
    digraph, n = {0: [0]}, 1  # Should return True
    res = solution.hasCycle(digraph, n)
    print(res)
    digraph, n = {0: [1], 1: []}, 2  # Should return False
    res = solution.hasCycle(digraph, n)
    print(res)
    digraph, n = {0: [1], 1: [0]}, 2  # Should return True
    res = solution.hasCycle(digraph, n)
    print(res)
    digraph, n = {0: [1], 2: [3], 3: []}, 4  # Should return False
    res = solution.hasCycle(digraph, n)
    print(res)
    digraph, n = {0: [1], 1: [2], 2: [0], 3: [4], 4: []}, 5  # Should return True
    res = solution.hasCycle(digraph, n)
    print(res)
    digraph, n = {0: [1], 1: [2], 3: []}, 4  # Should return False
    res = solution.hasCycle(digraph, n)
    print(res)
    digraph, n = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: [2]}, 5  # Should return True
    res = solution.hasCycle(digraph, n)
    print(res)

def cycleUndirected():
    solution = Solution()
    graph, n = {}, 0  # Should return False
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: []}, 1  # Should return False
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: [0]}, 1  # Should return True
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: [1], 1: [0]}, 2  # Should return False
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: [1, 2], 1: [0, 2], 2: [0, 1]}, 3  # Should return True
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}, 4  # Should return False
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2], 4: [5], 5: [4, 6], 6: [5, 4]}, 7  # Should return True
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: [1, 1], 1: [0, 0]}, 2  # Should return True
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4]}, 6  # Should return True
    res = solution.hasCycleUndirected(graph, n)
    print(res)
    graph, n = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}, 6  # Should return False
    res = solution.hasCycleUndirected(graph, n)
    print(res)

def bipartite():
    solution = Solution()
    graph, n = {}, 0 # Expected Output: True
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: []}, 1 # Expected Output: True
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: [1], 1: [0]}, 2, # Expected Output: True
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: [1, 2], 1: [0, 2], 2: [0, 1]}, 3 # Expected Output: False
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}, 4 # Expected Output: True
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: [1], 1: [0], 2: [3], 3: [2]}, 4 # Expected Output: True
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: [1], 1: [0], 2: [3, 4], 3: [2, 4], 4: [2, 3]}, 5 # Expected Output: False
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}, 5 # Expected Output: True
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: [0]}, 1 # Expected Output: False
    res = solution.isBipartite(graph, n)
    print(res)
    graph, n = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2], 4: [5], 5: [4, 6], 6: [5]}, 7 # Expected Output: True
    res = solution.isBipartite(graph, n)
    print(res)

def bcc():
    solution = Solution()
    graph, n = {}, 0 #  Expected Output: Bridges: [], Articulation Points: []
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: []}, 1 #  Expected Output: Bridges: [], Articulation Points: []
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: [1], 1: [0]}, 2 #  Expected Output: Bridges: [(0, 1)], Articulation Points: []
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: [1], 1: [0, 2], 2: [1]}, 3 #  Expected Output: Bridges: [(1, 2), (0, 1)], Articulation Points: [1]
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: [1, 2], 1: [0, 2], 2: [0, 1]}, 3 #  Expected Output: Bridges: [], Articulation Points: []
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: [1], 1: [0, 2], 2: [1], 3: [4], 4: [3]}, 5 #  Expected Output: Bridges: [(0, 1), (1, 2), (3, 4)], Articulation Points: [1]
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: [1, 4], 4: [3]}, 5 #  Expected Output: Bridges: [(1, 3), (3, 4)], Articulation Points: [1, 3]
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: [1], 1: [0, 2], 2: [1], 3: [4], 4: [3, 5], 5: [4]}, 6 #  Expected Output: Bridges: [(0, 1), (1, 2), (3, 4), (4, 5)], Articulation Points: [1, 4]
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}, 5 #  Expected Output: Bridges: [(0, 1), (0, 2), (0, 3), (0, 4)], Articulation Points: [0]
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)
    graph, n = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4]}, 6 #  Expected Output: Bridges: [(3,4),(4, 5)], Articulation Points: [3, 4]
    res, res2 = solution.findBCC(graph, n)
    print(res, res2)

def tsort():
    solution = Solution()
    graph, n = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}, 4  # Expected output: [0, 1, 2, 3]
    print(solution.tsort(graph, n))
    graph, n = {0: [1], 1: [2], 2: [3], 3: []}, 4  # Expected output: [0, 1, 2, 3]
    print(solution.tsort(graph, n))
    graph, n = {0: [1], 1: [2], 2: [0]}, 3  # Expected output: None (Cycle detected, no topological sort possible)
    print(solution.tsort(graph, n))
    graph, n = {0: [], 1: [0], 2: [1], 3: [1]}, 4  # Expected output: [3, 2, 1, 0] or [2, 3, 1, 0]
    print(solution.tsort(graph, n))
    graph, n = {0: [], 1: [2], 2: [3], 3: [4], 4: []}, 5  # Expected output: [1, 2, 3, 4, 0] or any valid topological order
    print(solution.tsort(graph, n))
    graph, n = {0: [1], 1: [], 2: [3], 3: [], 4: [1, 3]}, 5  # Expected output: [4, 0, 2, 3, 1] or any valid topological order    
    print(solution.tsort(graph, n))

if __name__ == '__main__':    
    digraphSearch()
    undirectedGraphSearch()
    cycleDirected()
    cycleUndirected()
    bipartite()
    bcc()
    tsort()
