""" Kruskal """
class UnionFind:
	def __init__(self, n):
		# every node is its own parent
		self.parents = list(range(n))
		self.counts = [1] * n
	
	# returns the root of the set that x belongs to
	def find(self, x):
		if self.parents[x] != x:
			# resetting the parent pointers to all members of the set to reduce amortized cost
			self.parents[x] = self.find(self.parents[x])
		return self.parents[x]
	
	def union(self, x, y):
		parent_x = self.find(x)
		parent_y = self.find(y)
		# whichever set is larger, that should be the root of the unioned set to reduce amortized cost
		if self.counts[parent_x] > self.counts[parent_y]:
			self.parents[parent_y] = parent_x
			self.counts[parent_x] += self.counts[parent_y]
		else:
			self.parents[parent_x] = parent_y
			self.counts[parent_y] += self.counts[parent_x]

def kruskal():
	# sort the edges
	# take each edge in sorted order
	# add it to the tree if the other end doesn't already belong to the same group
	# why do I need union find? why cannot we use a visited array?
	# the way the algorithm works is because it forms a disjointed set of trees and then connecting them
	# when connecting two disjointed sets, we need to ensure that we're not connecting within the same set
	# that would form a cycle
	disjoint_set = UnionFind(n + 1)
	total_cost = 0
	num_tree_edges = 0

	for u, v, cost in sorted(connections, key=lambda x:x[2]):
		if disjoint_set.find(u) != disjoint_set.find(v):
			disjoint_set.union(u, v)
			total_cost += cost
			num_tree_edges += 1
	
	return total_cost if num_tree_edges == n-1 else -1

def build(n, connections):
	adj = [[] for _ in range(n+1)]
	for u, v, cost in connections:
		adj[u].append((cost, v))
		adj[v].append((cost, u))
	return adj

""" Prim """
def prim():
	adj = build(n, connections)
	total_cost = 0
	num_tree_edges = 0
	# find all incident edges and mark them as candidates
	# find the candidate with minimum weight
	# go through the candidates from min weight to max one by one
	# if the edge doesn't lead to another node which is visited, use the edge
	# mark the vertex on the other end as visited so that we don't form circles

	# need a visited array to avoid cycles
	visited = [False] * (n + 1)

	# need a minheap for storing the candidate edges
	# need to store (cost, u, v) in the heap so that the mean is ordered by cost
	minheap = []

	# start from node 1
	visited[1] = True

	for cost, v in adj[1]:
		heapq.heappush(minheap, (cost, 1, v))

	# u should always be the one which is visited
	# if v is also visited, then we remove this edge
	while minheap:
		# pop because the minimum edge is always removed
		# either it forms part of the tree, or it is thrown away because adding it would form a cycle
		cost, u, v = heapq.heappop(minheap)
		if not visited[v]:
			total_cost += cost
			visited[v] = True
			num_tree_edges += 1

			for cost, w in adj[v]:
				heapq.heappush(minheap, (cost, v, w))

	return total_cost if num_tree_edges == n-1 else -1
