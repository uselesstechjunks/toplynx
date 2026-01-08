def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
	# the idea is to use binary search to find the range of matched words.
	# for the first letter in searchWord, we find the range [left, right] with
	# the match function. we add first three words within this range (if exists) to our
	# resultset.
	# 
	# then for the next letter, we confine ourselves within the existing range and find
	# a subrange for which the match only looks at the second letter between strings
	#
	# we repeat this process until the range diminshes or we run out of letters
	# we need to sort the products list beforehand to be able to do binary search on it.
	#
	# note that to be able to find the range, we need to convert the condition which 
	# extends the range from either ends.
	def eq(str1, str2, k):
		# need to ensure that we're not comparing shorter words
		if k >= len(str1) or k >= len(str2):
			return False
		return str1[k] == str2[k]

	def geq(str1, str2, k):
		# need to ensure that we're not comparing shorter words
		if k >= len(str1) or k >= len(str2):
			return False
		return str1[k] <= str2[k]

	def leq(str1, str2, k):
		# need to ensure that we're not comparing shorter words
		if k >= len(str1) or k >= len(str2):
			return False
		return str1[k] >= str2[k]

	def find_left(words, left, right, word, k):
		# find the leftmost index where words[index][k] <= word[k]
		while left != right:
			mid = (left + right) // 2
			if leq(words[mid], word, k):
				right = mid
			else:
				left = mid + 1
		return left if eq(words[left], word, k) else -1

	def find_right(words, left, right, word, k):
		# find the leftmost index where word[k] <= words[index][k]
		while left <= right:
			mid = (left + right) // 2
			if geq(words[mid], word, k):
				left = mid + 1
			else:
				right = mid - 1
		return right if right >= 0 and eq(words[right], word, k) else -1

	products.sort()
	left, right = 0, len(products)-1
	results = []

	for k in range(len(searchWord)):
		if left <= right and left > -1:
			left = find_left(products, left, right, searchWord, k)
			right = find_right(products, left, right, searchWord, k)
			end_list = min(left + 3, right + 1)
			results.append(products[left:end_list])
		else:
			results.append([])
	return results