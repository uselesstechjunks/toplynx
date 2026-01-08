def maximalSquare(self, matrix: List[List[str]]) -> int:
	# if the current entry in the matrix is 1, then it can contribute
	# to 3 possible squares, one ending with top (i-1, j), one on the left
	# (i, j-1) and one ending on the corner (i-1, j-1)
	# case by case explanation:
	# if the square ending at (i-1, j) is the min, then we know for sure
	# that the one ending at (i, j-1) covers at least as many rows as (i-1, j). 
	# this is guaranteed since they are SQUARES not RECTANGLES.
	m, n = len(matrix), len(matrix[0]) if len(matrix) > 0 else 0
	prev_row = [1 if x == '1' else 0 for x in matrix[0]]
	curr_row = [0] * n
	result = max(prev_row)
	for i in range(1, m):
		curr_row[0] = 1 if matrix[i][0] == '1' else 0
		result = max(result, curr_row[0])
		for j in range(1, n):
			if matrix[i][j] == '1':
				curr_row[j] = min(curr_row[j-1], prev_row[j-1], prev_row[j]) + 1
				result = max(result, curr_row[j])
		for j in range(n):
			prev_row[j] = curr_row[j]
			curr_row[j] = 0
	return result * result
	
def maximalRectangle(self, matrix: List[List[str]]) -> int:
	# the approach for maximal square doesn't apply here
	# we solve it by folding it and forming histograms
	def max_consecutive_ones(arr):
		left = 0
		res = 0
		for right in range(len(arr)):
			if arr[right] == 0:
				left = right + 1
			else:
				res = max(right - left + 1, res)
		return res

	def largest_histogram(heights):
		# we scan from left to right and check the min of every interval
		# everything that gets popped is the min of some interval
		max_area = 0
		minstack = [-1]
		for i, height in enumerate(heights):
			while minstack[-1] != -1 and heights[minstack[-1]] >= height:
				prev_min_index = minstack.pop()
				# prev_min_index is the min of the range(stack[-1] + 1, i - 1)
				left, right = minstack[-1] + 1, i - 1 # IMPORTANT: THIS IS WHY WE NEED -1
				width = right - left + 1
				max_area = max(max_area, width * heights[prev_min_index])
			minstack.append(i)
		# everything that remains on stack is also min of some range
		while minstack[-1] != -1:
			prev_min_index = minstack.pop()
			# prev_min_index is the min of the range(stack[-1] + 1, n - 1)
			left, right = minstack[-1] + 1, len(heights) - 1
			width = right - left + 1
			max_area = max(max_area, width * heights[prev_min_index])
		return max_area

	m, n = len(matrix), len(matrix[0]) if len(matrix) > 0 else 0
	if not m or not n:
		return 0

	histogram = [1 if x == '1' else 0 for x in matrix[0]]
	max_area = max_consecutive_ones(histogram)
	for i in range(1, m):
		for j in range(n):
			histogram[j] = histogram[j] + 1 if matrix[i][j] == '1' else 0
		max_area = max(max_area, largest_histogram(histogram))
	return max_area
