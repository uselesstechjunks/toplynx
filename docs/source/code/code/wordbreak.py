def wordBreak(self, s: str, wordDict: List[str]) -> bool:
	def good_approach():
		""" O(n*m*k) """
		# f(i) := s[0...i] can be segmented or not
		# f(i) = true of any f(j-1, j <= i) and s[j,i] in in dict
		words = set(wordDict)
		n = len(s)
		dp = [False] * n

		for i in range(n):
			for word in words:
				size = len(word)
				j = i-size+1
				if (j == 0 or (j > 0 and dp[j-1])) and (j >= 0 and s[j:i+1] == word):
					dp[i] = True
					break

		return dp[-1]

	def bad_approach():
		""" O(n^2) """
		# f(i) := s[0...i] can be segmented or not
		# f(i) = true of any f(j-1, j <= i) and s[j,i] in in dict
		words = set(wordDict)
		n = len(s)
		dp = [False] * n

		for i in range(n):
			for j in range(i+1):
				if (j == 0 or dp[j-1]) and s[j:i+1] in words:
					dp[i] = True
					break
		
		return dp[-1]
	
	return good_approach()

def wordBreakII(self, s: str, wordDict: List[str]) -> List[str]:
	def backtrack(curr, start_index):
		nonlocal result
		n = len(s)
		if start_index >= n:
			ans = ' '.join(curr)
			result.append(ans)
			return
		
		for j in range(start_index, n):
			if is_word(start_index, j):
				size = j - start_index + 1
				word = s[start_index: start_index + size]
				curr.append(word)
				backtrack(curr, start_index + size)
				curr.pop()
	
	def is_word(i, j):
		nonlocal words
		# curr = s[i:j+1]
		# return curr in words
	

	words = set(wordDict)
	n = len(s)
	# if dp[i][j] = True => s[i...j] is a valid word in the dict
	dp = [[False] * n for _ in range(n)]

	# TODO implement using trie
	print(dp)
	result = []
	backtrack([], 0)
	return result
