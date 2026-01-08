from functools import cache

def findTargetSumWays(self, nums: List[int], target: int) -> int:
	"""
	IMPORTANT!! NEED TO KEEP IN MIND THAT RUNNING SUM HAS TO BE IN THE STATE.
	WE CANNOT MEMOIZE/USE DP IF WE KEEP THE RECURSIVE CALL STRUCTURE REDUCING
	TARGET. THAT METHOD IS NOT STATELESS AS COUNT CHANGES ACROSS CALLS.
	"""
	def bounded_knapsack():
		# f(i, j) = number of ways to obtain target j after seeing i nums
		# f(0, 0) = 1
		# f(i+1, j) = f(i, j-nums[i]) + f(i, j+nums[i])
		# update rule: 
		# every f(i, j) would update f(i+1, j-nums[i]) and f(i+1, j+nums[i])
		""" 
		This is super important because if we loop the other way, we'd miss out on many updates.
		"""
		# range for j: [-sum(abs(nums)), sum(abs(nums))]
		# range for i: 0...n-1
		# gotta be careful with offset
		n = len(nums)
		max_sum = sum(nums)
		if target > max_sum or target < -max_sum:
			return 0
		max_range = 2 * max_sum + 1
		dp = [[0] * max_range for _ in range(n + 1)]
		# dp[-][max_sum] actually represents curr sum = 0
		dp[0][max_sum] = 1 # important
		for i in range(n):
			"""
			Incorrect way of doing it:
			for j in range(nums[i], max_range - nums[i]):
				dp[i+1][j] = dp[i][j-nums[i]] + dp[i][j+nums[i]]
			"""
			# deciding on the range of j
			for j in range(nums[i], max_range - nums[i]):
				dp[i+1][j-nums[i]] += dp[i][j]
				dp[i+1][j+nums[i]] += dp[i][j]
		return dp[-1][max_sum + target]

	def memoized_with_annotation():
		@cache
		def dfs(index, curr_sum):
			if index == len(nums):
				return 1 if curr_sum == target else 0
			sub_curr = dfs(index + 1, curr_sum - nums[index])
			add_curr = dfs(index + 1, curr_sum + nums[index])
			return sub_curr + add_curr
		return dfs(0, 0)

	def memoized():
		# the idea is - every time we reach target 0 after seeing n numbers, we increase the count
		# memo dimension: index X max_range
		# note that the state here is current_sum as opposed to target
		# the reason is - to make the function calls stateless, we cannot use
		# reduced target as every time we call the same target the overall count
		# increases. on the other hand, current sum with the same index remains same
		# and hence can be memoized
		# range of curr_sum: 2*max_sum + 1
		# max_sum = 10, range: [-10,...-1, 0, 1,...10]
		#               index: [  0,    9,10,11,   21]
		max_sum = sum([abs(num) for num in nums])
		max_range = 2 * max_sum + 1
		memo = [[-1] * max_range for _ in range(len(nums))]

		def dfs(index, curr_sum):
			# not storing index=n entries because we only need 1
			if index == len(nums):
				return 1 if curr_sum == target else 0
			curr_sum_index = curr_sum + max_sum
			if memo[index][curr_sum_index] == -1:
				sub_curr = dfs(index + 1, curr_sum - nums[index])
				add_curr = dfs(index + 1, curr_sum + nums[index])
				memo[index][curr_sum_index] = sub_curr + add_curr
			return memo[index][curr_sum_index]

		return dfs(0, 0)

	def recursive_stateless():
		def dfs(index, curr_sum):
			if index == len(nums):
				return 1 if curr_sum == target else 0
			sub_curr = dfs(index + 1, curr_sum - nums[index])
			add_curr = dfs(index + 1, curr_sum + nums[index])
			return sub_curr + add_curr
		return dfs(0, 0)

	def recursive_with_external_variable():
		count = 0
		def dfs(index, curr_sum):
			nonlocal count
			if index == len(nums): # reached the end
				if curr_sum == target: # achieved the target
					count += 1
			else:
				dfs(index + 1, curr_sum - nums[index])
				dfs(index + 1, curr_sum + nums[index])
				# no need to do anything here as the state is updated on
				# reaching the target to the external variable
		dfs(0, 0)
		return count
	return bounded_knapsack()
