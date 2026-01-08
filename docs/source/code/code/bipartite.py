def isBipartite(self, graph: List[List[int]]) -> bool:
	def compliment(colour):
		# returns 1 when colour = 0
		# returns 0 when colour = 1
		return 1 - colour

	def using_bfs():
		n = len(graph)
		colours = [2] * n # colour 2:= unvisited

		def bfs(node):
			nonlocal colours
			queue = deque([node])

			while queue:
				node = queue.popleft()
				for child in graph[node]:
					if colours[child] == colours[node]:
						return False
					if colours[child] == 2:
						colours[child] = compliment(colours[node])
						queue.append(child)
			return True
		
		for node in range(n):
			if colours[node] == 2:
				colours[node] = 0
				if not bfs(node):
					return False
		
		return True

	def using_dfs():
		n = len(graph)
		colours = [2] * n # colour 2:= unvisited

		def dfs(node, parent):
			nonlocal colours
			
			for child in graph[node]:
				if colours[child] == colours[node]:
					return False
				if child != parent and colours[child] == 2:
					colours[child] = compliment(colours[node])
					if not dfs(child, node):
						return False

			return True
		
		for node in range(n):
			if colours[node] == 2:
				colours[node] = 0
				if not dfs(node, -1):
					return False
		
		return True