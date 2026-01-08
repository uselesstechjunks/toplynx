class Solution:
    def bcc(self, n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u-1].append(v-1)
            adj[v-1].append(u-1)

        discovered = [False] * n
        depth = [float('inf')] * n
        low = list(range(n))
        res = set()

        def dfs(adj, discovered, low, depth, d, parent, u, res):
            discovered[u] = True
            depth[u] = d
            isArticulation = False
            treedeg = 0

            for v in adj[u]:
                if not discovered[v]:
                    treedeg += 1
                    dfs(adj, discovered, low, depth, d + 1, u, v, res)
                    if low[v] >= depth[u]:
                        isArticulation = True
                    low[u] = min(low[u], low[v])
                elif v != parent:
                    low[u] = min(low[u], depth[v])

            if (parent != -1 and isArticulation) or (parent == -1 and treedeg > 1):
                res.add(u)

        for u in range(n):
            if not discovered[u]:
                dfs(adj, discovered, low, depth, 0, -1, u, res)

        print(discovered)
        print(depth)
        print(low)
        print(res)

        return None, None

def test1():
    solution = Solution()
    n = 5  
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (1, 3)]
    cutnodes, bcc = solution.bcc(n, edges)
    print(cutnodes)
    print(bcc)

def test2():
    solution = Solution()
    n = 4
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    cutnodes, bcc = solution.bcc(n, edges)
    print(cutnodes)
    print(bcc)

def test3():
    solution = Solution()
    n = 6
    edges = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4)]
    cutnodes, bcc = solution.bcc(n, edges)
    print(cutnodes)
    print(bcc)

def test4():
    solution = Solution()
    n = 6
    edges = [(1, 2), (2, 3), (4, 5)]
    cutnodes, bcc = solution.bcc(n, edges)
    print(cutnodes)
    print(bcc)

def test5():
    solution = Solution()
    n = 7
    edges = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 7), (7, 4)]
    cutnodes, bcc = solution.bcc(n, edges)
    print(cutnodes)
    print(bcc)

def test6():
    solution = Solution()
    n = 1
    edges = []
    cutnodes, bcc = solution.bcc(n, edges)
    print(cutnodes)
    print(bcc)

def test7():
    solution = Solution()
    n = 4
    edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    cutnodes, bcc = solution.bcc(n, edges)
    print(cutnodes)
    print(bcc)

if __name__ == '__main__':
    #test1()
    #test2()
    #test3()
    #test4()
    test5()
    #test6()
    #test7()