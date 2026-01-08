def orangesRotting(self, grid: List[List[int]]) -> int:
	m, n = len(grid), len(grid[0]) if len(grid) > 0 else 0

	queue = deque()
	fresh_count = 0

	for i in range(m):
		for j in range(n):
			if grid[i][j] == 2:
				queue.append((i,j))
			elif grid[i][j] == 1:
				fresh_count += 1

	directions = [(-1,0),(1,0),(0,-1),(0,1)]

	def next_direction(x, y, direction):
		return x + direction[0], y + direction[1]

	def within_boundaries(x, y):
		nonlocal m, n
		return 0 <= x and x < m and 0 <= y and y < n

	steps = 0

	while queue and fresh_count:
		size = len(queue)
		for _ in range(size):
			i, j = queue.popleft()
			for direction in directions:
				x, y = next_direction(i, j, direction)
				if within_boundaries(x, y) and grid[x][y] == 1:
					fresh_count -= 1
					grid[x][y] = 2
					queue.append((x, y))
		steps += 1

	return steps if not fresh_count else -1