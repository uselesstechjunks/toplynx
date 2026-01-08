# with cooldown
def maxProfit(self, prices: List[int]) -> int:
	"""
	states: (a) not holding (b) holding (c) cooling down
	state transition:
	1. not holding:
		(a) buy -> holding
		(b) nothing -> not holding
	2. holding:
		(a) sell -> cooling down
		(b) nothing -> holding
	3. cooling down:
		(a) nothing -> not holding
	"""
	n = len(prices)
	dp = [[0] * 3 for _ in range(n)]
	dp[0][1] = -prices[0]
	print(dp[0])
	for i in range(1, n):
		# can come to not holding state from (a) not holding and (b) cooling down
		dp[i][0] = max(dp[i-1][0], dp[i-1][2])
		# can come to holding state from (a) not holding + buy (b) holding
		dp[i][1] = max(dp[i-1][0] - prices[i], dp[i-1][1])
		# can come to cooling down state from holding state
		dp[i][2] = dp[i-1][1] + prices[i]
		print(i, prices[i], dp[i])
	return max(dp[-1])

# with at most 2 transactions
def maxProfit(self, prices: List[int]) -> int:
	"""
	states:
		1. no transactions
		2. one transaction and holding
		3. one transaction and not holding
		4. two transactions and holding
		5. two transactions and not holding
	"""
	n = len(prices)
	""" NOTE -inf instead of 0 """
	dp = [[-inf] * 5 for _ in range(n)]
	dp[0][1] = -prices[0]
	""" NOTE this initialization is required because of -inf """
	dp[0][0] = 0
	for i in range(1, n):
		dp[i][0] = dp[i-1][0]
		dp[i][1] = max(dp[i-1][0] - prices[i], dp[i-1][1])
		dp[i][2] = max(dp[i-1][1] + prices[i], dp[i-1][2])
		dp[i][3] = max(dp[i-1][2] - prices[i], dp[i-1][3])
		dp[i][4] = max(dp[i-1][3] + prices[i], dp[i-1][4])
	return max(dp[-1])
