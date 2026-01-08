from collections import deque

def kahn():
	# won't work if there are cycles
	def build(n, edges):
		in_degrees = [0] * n
		adj = [[] for _ in range(n)]
		for a, b in edges:
			adj[b].append(a)
			in_degrees[a] += 1
		return adj, in_degrees
	
	def bfs():
		nonlocal adj, visited, queue, tsorted, in_degrees
		while queue:
			u = queue.popleft()
			tsorted.append(u)
			for v in adj[u]:
				in_degrees[v] -= 1
				if not in_degrees[v]:
					queue.append(v)

	adj, in_degrees = build(numCourses, prerequisites)
	visited = [False] * numCourses

	queue = deque([])
	for u in range(numCourses):
		if not in_degrees[u]:
			visited[u] = True
			queue.append(u)

	tsorted = []
	if bfs():
		return []
	return tsorted if len(tsorted) == numCourses else []

def tarjan():
	# since this uses dfs, it can detect cycles
	def build(n, edges):
		adj = [[] for _ in range(n)]
		for a, b in edges:
			adj[b].append(a)
		return adj

	def has_cycle(u):
		nonlocal visited, finished, adj, tsorted
		visited[u] = True
		for v in adj[u]:
			if visited[v] and not finished[v]: # back edge
				return True # cycle exists
			if not visited[v] and has_cycle(v):
				return True
		finished[u] = True
		tsorted.insert(0, u)
		return False

	adj = build(numCourses, prerequisites)
	tsorted = []
	visited = [False] * numCourses
	finished = [False] * numCourses
	# no need to find root edges in the dag
	# dfs would figure it out
	for u in range(numCourses):
		if not visited[u]:
			# dfs returns true if there is a cycle, in which case it's not a dag
			# we return empty list
			if has_cycle(u):
				return []
	return tsorted
