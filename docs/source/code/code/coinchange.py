def coinChange(self, coins: List[int], amount: int) -> int:
	def unbounded_knapsack():
		# f(i, j) = min number of coins from the set coins[0...i] to make up for amount j
		# transition rule
		# f(i, j) = min(f(i-1, j), f(i, j-coins[i]) + 1)
		""" 
		if appears that we are only considering the case where we take coins[i] just once.
		but when we PROCESS THIS LEFT TO RIGHT, so, if f(i, j-k*coins[i]) really was
		the optimal ans for something we'd have considered it.
		
		!!!!!IMPORTANT!!!!!
		when the state (i,j) is updated, all (i,k<j) states should have been updated with
		the knowledge that coins[i] exists in this world!
		"""
		# removing the first dimension
		# f(j) = min(f(j), f(j-coins[i]) + 1)
		n = len(coins)
		dp = [inf] * (amount + 1)
		dp[0] = 0
		for coin in coins:
			for j in range(coin, amount + 1):
				dp[j] = min(dp[j], dp[j-coin] + 1)
		return dp[-1] if dp[-1] < inf else -1

	def bfs_bottom_up():
		# the idea is to start traversing on the state transition
		# graph. start node is amount, and destination node is 0.
		# for each node in this graph, we have coins outgoing edges
		# O(V+E) = O(S+S*N)
		queue = deque([0])
		visited = set([])
		steps = 0
		while queue:
			size = len(queue)
			for _ in range(size):
				curr_amount = queue.popleft()
				if curr_amount == amount:
					return steps
				for coin in coins:
					new_amount = curr_amount + coin
					if new_amount <= amount and new_amount not in visited:
						queue.append(new_amount)
						visited.add(new_amount)
			steps += 1
		return -1
	def bfs_top_down():
		# the idea is to start traversing on the state transition
		# graph. start node is amount, and destination node is 0.
		# for each node in this graph, we have coins outgoing edges
		# O(V+E) = O(S+S*N)
		queue = deque([amount])
		visited = set([])
		steps = 0
		while queue:
			size = len(queue)
			for _ in range(size):
				curr_amount = queue.popleft()
				if curr_amount == 0:
					return steps
				for coin in coins:
					remaining_amount = curr_amount - coin
					if remaining_amount >= 0 and remaining_amount not in visited:
						queue.append(remaining_amount)
						visited.add(remaining_amount)
			steps += 1
		return -1

def coinChangeCountWays(self, amount: int, coins: List[int]) -> int:
	""" Note: f(i,j) = f(i-1,j) + f(i,j-coins[i]) """
	# f(i-1,j) := don't use j-th coin
	# f(i,j-coins[i]) := use j-th coin
	dp = [0] * (amount + 1)
	for coin in coins:
		dp[0] = 1
		for j in range(coin, amount + 1):
			dp[j] += dp[j-coin]
	return dp[-1]
