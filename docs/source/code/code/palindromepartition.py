def partition(self, s: str) -> List[List[str]]:
	# the idea is the find all existing palindrome lengths using dp
	# f(i,j) := length of palindrome if s[i+1...j-1] is a palindrome, 0 otherwise
	#           j-i-1 or 0
	# transition rule:
	# f(i,j) = f(i+1,j-1) + 2 if s[i] = s[j] and f(i+1,j-1) > 0, 0 otherwise
	# for computing the value of (i,j), we need next row and previous column.
	# so we need to fill it from bottom to top, left to right.
	# once we have the table, we can form the parition using backtracking
	def backtrack(curr, start_index):
		"""
		time complexity:
		number of calls = number of branches in partition tree = O(2^n)
		for each call, palindrome check O(1), copy string O(n)
		total: O(n*2^n)
		"""
		nonlocal result, n, counts
		if start_index >= n:
			result.append(curr[:])
			return

		# loop over all possible lengths of palindromes starting from start_index
		# add palindromic strings to curr and backtrack from next index
		for j in range(start_index, n):
			# print(f'checking substr[{start_index}:{j}]')
			counts[(start_index,j)] += 1
			if is_palindrome(start_index, j):
				size = j - start_index + 1
				palindrome = s[start_index: start_index + size]
				curr.append(palindrome)
				backtrack(curr, start_index + size)
				curr.pop()

	def is_palindrome(i, j):
		nonlocal dp
		return dp[i][j]
		# curr = s[i:j+1]
		# return curr == curr[::-1]
	
	def find_palindromes():
		# complexity: O(n^2)
		nonlocal dp, n
		for i in range(n-1, -1, -1): # O(n)
			for j in range(i+1, n): # O(n)
				if s[i] != s[j] or not dp[i+1][j-1]:  # O(1)
					dp[i][j] = False # O(1)

	n = len(s) # O(1)
	dp = [[True] * n for _ in range(n)] # O(n^2)

	counts = defaultdict(int)
	
	find_palindromes() # O(n^2)

	result = [] # O(1)
	backtrack([], 0) # ??

	# print(counts)
	# print(f'number of substrings:{len(counts.keys())}')
	# print(f'number of calls:{sum(counts.values())}')
	
	return result
