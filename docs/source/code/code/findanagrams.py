def findAnagrams(self, s: str, p: str) -> List[int]:
	counts = Counter(p)
	mismatch = len(counts)
	k = len(p)
	result = []
	for index in range(len(s)):
		if s[index] in counts:
			counts[s[index]] -= 1
			if counts[s[index]] == 0:
				mismatch -= 1
			elif counts[s[index]] == -1:
				mismatch += 1
		if k <= index and s[index-k] in counts:
			counts[s[index-k]] += 1
			if counts[s[index-k]] == 1:
				mismatch += 1
			elif counts[s[index-k]] == 0:
				mismatch -= 1
		if mismatch == 0:
			result.append(index - k + 1)
	return result