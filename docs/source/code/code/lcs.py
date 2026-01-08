def lcs():
	# say, we have sene i characters from text1 and j characters from text2
	# the lcs(i, j) contains the longest common subsequence.
	# if text1[i+1] == text2[j+1] then we increase lcs by 1
	# else lcs is the max of lcs(i, j), lcs(i+1, j) and lcs(i, j+1)
	# in the end, the final lcs is found at lcs(m, n)
	m, n = len(text1), len(text2)
	# dp table
	lcs = [[0] * (n+1) for _ in range(m+1)]
	# for every lcs(i+1, j+1), we need lcs(i, j), lcs(i+1, j) and lcs(i, j+1)
	# that is, previous row and previous column of the same row
	# so we have to fill the array from left to right, top to bottom
	for i in range(m):
		for j in range(n):
			if text1[i] == text2[j]:
				lcs[i+1][j+1] = lcs[i][j] + 1
			else:
				lcs[i+1][j+1] = max(lcs[i+1][j], lcs[i][j+1])

	return lcs[m][n]

def lcs_memory_optimized():
	# f(i,j) = length of lcs from s[0..i-1] and t[0...j-1]
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

def construct(stack, indicators, i, j):
	if i == 0 or j == 0:
		return
	if indicators[i][j] == 1:
		construct(stack, indicators, i-1, j-1)
		# need to keep this in mind that we only add characters in this case
		stack.append(text1[i-1])
	elif indicators[i][j] == 2:
		construct(stack, indicators, i-1, j)
	else:
		construct(stack, indicators, i, j-1)

def lcs_str() -> str:
	# need to keep additional storage for longest common subsequence string
	# just need to store an indicator:
	# 0 means it came from same row, previous column
	# 1 means it came from previous row, previous column
	# 2 means it came from previous row, same column
	# also need to keep track of the index??
	m, n = len(text1), len(text2)
	prev_row, curr_row = [0] * (n+1), [0] * (n+1)
	indicators = [[0] * (n+1) for _ in range(m+1)]

	for i in range(m):
		for j in range(n):
			if text1[i] == text2[j]:
				curr_row[j+1] = prev_row[j] + 1
				indicators[i+1][j+1] = 1
			else:
				if prev_row[j+1] < curr_row[j]:
					curr_row[j+1] = curr_row[j]
					indicators[i+1][j+1] = 0
				else:
					curr_row[j+1] = prev_row[j+1]
					indicators[i+1][j+1] = 2

		# copy current row to previous
		for k in range(len(curr_row)):
			prev_row[k] = curr_row[k]

	stack = []
	construct(stack, indicators, m, n)
	return ''.join(stack)
