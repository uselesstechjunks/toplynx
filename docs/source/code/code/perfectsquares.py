def numSquares(self, n: int) -> int:
	def knapsack_optimized():
		# f(i,j) := min perfect squares numbers [1..i^2] to make up sum j
		# transition rule:
		# f(i,j) = min f(i-1,j), f(i,j-i^2)+1
		# dimension i can be reduced (as long as it's kept as a variable)
		# f(j) = min f(j), f(j-i^2) + 1
		# NOTE when updating the state (i,j), we need j-i*2 from current iteration
		# so we need to fill it from left to right
		m = floor(sqrt(n)) + 1
		dp = [inf] * (n+1)
		# 0 numbers to make up sum 0
		for i in range(1,m):
			dp[0] = 0
			for j in range(1,n+1):
				if j >= i*i:
					dp[j] = min(dp[j], dp[j-i*i] + 1)
		return dp[-1]
	def knapsack():
		# f(i,j) := min perfect squares numbers [1..i^2] to make up sum j
		# transition rule:
		# f(i,j) = min f(i-1,j), f(i,j-i^2)+1
		m = floor(sqrt(n)) + 1
		dp = [[inf] * (n+1) for _ in range(m)]
		# 0 numbers to make up sum 0
		for i in range(m):
			dp[i][0] = 0
		for i in range(1,m):
			for j in range(1,n+1):
				dp[i][j] = dp[i-1][j]
				if j >= i*i:
					dp[i][j] = min(dp[i][j], dp[i][j-i*i] + 1)
		return dp[-1][-1]

	def bfs():
		pass