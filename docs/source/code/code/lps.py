def longestPalindromeSubseq(self, s: str) -> int:
	def lps(s):
		# f(i,j) = length of lps after seeing s[i+1...j-1]
		# need to compute f(i-1,j+1) after seeing s[i] and s[j]
		# transition rule:
		# f(i-1,j+1) = f(i,j) + 2 if s[i] == s[j]
		# f(i-1,j+1) = max(f(i-1,j), f(i,j+1)) otherwise
		# normalized equations
		# f(i,j) = f(i+1,j-1) + 2 if s[i] == s[j]
		# f(i,j) = max(f(i,j-1), f(i+1,j)) otherwise
		# need to fill rows from bottom to top
		# cannot remove dimension i
		# when state (i,j) is updated, we need to know
		#   j-1 from next row
		#   j-1 from current row and j from next row
		# so we need to keep track of the previous row
		n = len(s)
		dp_prev, dp = [0] * n, [0] * n
		for i in range(n-1, -1, -1):
			dp[i] = 1 # important because s[i..i] is a palindrome
			for j in range(i+1, n):
				if s[i] == s[j]:
					dp[j] = dp_prev[j-1] + 2
				else:
					dp[j] = max(dp[j-1], dp_prev[j])
			dp_prev = dp[:]
		return dp[-1]

	def lcs(s, t):
		# f(i,j) = length of lcs from s[0..i-1] and t[0...j-1]
		# need to compute f(i+1,j+1) after seeing s[i] and t[j]
		# transition rule:
		# f(i+1,j+1) = f(i,j) + 1 if s[i] = t[j]
		# f(i+1,j+1) = max(f(i+1,j), f(i,j+1))
		""" IMPORTANT THIS """
		# cannot be reduced to one dimension
		# f(j+1) = f(j) + 1 if s[i] == t[j] (need f(j) from previous iteration)
		# f(j+1) = f(j) (current iteration) + f(j+1) (previous iteration)
		# need to keep track of the previous row
		m, n = len(s), len(t)
		dp_prev, dp = [0] * (n + 1), [0] * (n + 1)
		for i in range(m):
			for j in range(n):
				if s[i] == t[j]:
					dp[j+1] = dp_prev[j] + 1
				else:
					dp[j+1] = max(dp[j], dp_prev[j+1])
			dp_prev = dp[:]
		return dp[-1]

def longestPalindromeSubstr(self, s: str) -> int:
	def lps(s):
		# f(i,j) = length of lps after seeing s[i+1...j-1]
		# need to compute f(i-1,j+1) after seeing s[i] and s[j]
		# transition rule:
		# f(i-1,j+1) = f(i,j) + 2 if s[i] == s[j] and f(i,j) = len(s[i+1...j-1])
		# f(i-1,j+1) = max(f(i-1,j), f(i,j+1)) otherwise
		# normalized equations
		# f(i,j) = f(i+1,j-1) + 2 if s[i] == s[j]
		# f(i,j) = max(f(i,j-1), f(i+1,j)) otherwise
		# need to fill rows from bottom to top
		# cannot remove dimension i
		# when state (i,j) is updated, we need to know
		#   j-1 from next row
		#   j-1 from current row and j from next row
		# so we need to keep track of the previous row
		n = len(s)
		dp_prev, dp = [0] * n, [0] * n
		start, size = 0, 1
		for i in range(n-1, -1, -1):
			dp[i] = 1
			for j in range(i+1, n):
				if s[i] == s[j] and dp_prev[j-1] == j-1-i:
					dp[j] = dp_prev[j-1] + 2
					if dp[j] > size:
						start = i
						size = dp[j]
				else:
					dp[j] = max(dp[j-1], dp_prev[j])
			dp_prev = dp[:]
		return s[start:start + size]
	return lps(s)
