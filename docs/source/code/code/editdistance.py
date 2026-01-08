def minEditDistance(self, word1: str, word2: str) -> int:
	# f(i,j) = edit distance after seeing s[0...i-1] and t[0...j-1]
	# f(i+1, j+1) = f(i, j) if s[i] == t[j]
	# else
	""" 
	Trick: 
		(1) Think of the next character we'd look after each operation.
		(2) Normalize the equations afterwards
	"""
	# replace: f(i+1,j+1) = f(i,j) + 1
	# delete:  f(i+1,j) = f(i,j) + 1 => f(i+1,j+1) = f(i,j+1) + 1
	# insert:  f(i, j+1) = f(i,j) + 1 => f(i+1,j+1) = f(i+1,j) + 1
	"""
	Final equations:
		f(i+1,j+1) = f(i,j) if s[i] == s[j] else (min(f(i,j),f(i,j+1),f(i+1,j))) + 1
	
	Cannot be reduced to lower dimensions - need both (i,j) and (i+1,j) at the same time
	"""
	m, n = len(word1), len(word2)
	dp = [[0] * (n+1) for _ in range(m+1)]
	""" Key: IMPORTANT TO INITIALIZE PROPERLY """
	for i in range(m + 1):
		dp[i][0] = i # all delete
	for j in range(n + 1):
		dp[0][j] = j # all insert
	for i in range(m):
		for j in range(n):
			if word1[i] == word2[j]:
				dp[i+1][j+1] = dp[i][j]
			else:
				dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1
	return dp[-1][-1]
