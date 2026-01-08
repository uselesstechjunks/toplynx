def uniquePaths(self, m: int, n: int) -> int:
	def grid():
		dp = [[0] * n for _ in range(m)]
		for i in range(m):
			dp[i][0] = 1
		for j in range(n):
			dp[0][j] = 1
		for i in range(1, m):
			for j in range(1, n):
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
		return dp[-1][-1]
	
	def optimized():
		# f(i,j) = f(i-1,j) + f(i,j-1)
		# remove the first diemsion
		# f(j) = f(j) + f(j-1) <- as long as f(j-1) is from the current iteration
		# and f(j) is from the previous iteration
		# so we need to fill this from left to right
		dp = [1] * n
		for i in range(1, m):
			for j in range(1, n):
				dp[j] = dp[j] + dp[j-1]
		return dp[-1]
	
	return optimized()

def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
	# dp[i][j] = 0 if obstacle
	# dp[i][j] = dp[i-1][j] + dp[i][j-1]
	# we can remove first dimension as it only depends on i-1
	# dp[j] = dp[j] (from previous row) + dp[j-1] (current row)
	# we need to fill it from left to right
	# initialization: if there is an obstacle in the first row,
	# all other cells after that would be unreachable
	m, n = len(obstacleGrid), len(obstacleGrid[0]) if len(obstacleGrid) > 0 else 0
	dp = [0] * n
	j = 0
	while j < n and not obstacleGrid[0][j]:
		dp[j] = 1
		j += 1
	for i in range(1, m):
		# this step is very important
		# we not only need to check for obstacles in the current cell, but also previous cell
		dp[0] = 1 if not obstacleGrid[i][0] and dp[0] else 0
		for j in range(1, n):
			dp[j] = dp[j] + dp[j-1] if not obstacleGrid[i][j] else 0
	return dp[-1]
