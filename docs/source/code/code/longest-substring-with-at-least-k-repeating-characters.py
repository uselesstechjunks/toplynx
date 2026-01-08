class Solution:
	def longest(self, s, max_distinct, min_repeat):
		""" 
		returns longest substring with 
		- at most 'max_distinct' distinct characterrs
		- each repeating >= min_repeat times
		"""
		n = len(s)
		counts = defaultdict(lambda:0)
		max_len = 0
		# distinct_key_count: stores the number of distinct keys
		# valid_key_count: stores the number of keys repeating >= k times
		distinct_key_count, valid_key_count = 0, 0
		l = 0
		for r in range(n):
			counts[s[r]] += 1
			if counts[s[r]] == 1:
				distinct_key_count += 1
			if counts[s[r]] == min_repeat:
				valid_key_count += 1
			while distinct_key_count > max_distinct and l <= r:
				counts[s[l]] -= 1
				if counts[s[l]] == 0:
					distinct_key_count -= 1
				if counts[s[l]] == min_repeat-1:
					valid_key_count -= 1
				l += 1
			if distinct_key_count == valid_key_count:
				max_len = max(max_len, r-l+1)
		return max_len
	def longestSubstring(self, s: str, k: int) -> int:		
		total_distinct = len(set(s))
		max_len = 0
		for max_distinct in range(1, total_distinct + 1):
			max_len = max(max_len, self.longest(s, max_distinct, k))
		return max_len