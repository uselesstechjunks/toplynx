def sortArray(self, nums: List[int]) -> List[int]:
	# bubble sort
	for i in range(1, len(nums)):
		j = i-1
		while j >= 0 and nums[j+1] < nums[j]:
			nums[j+1], nums[j] = nums[j], nums[j+1]
			j -= 1

	# insertion sort
	# iterate through every number from left to right
	# assume that every number until now is already sorted
	# loop through the left part moving every element towards right by 1
	# as long as we don't find something smaller
	# at this state, insert at the next position
	for i, num in enumerate(nums):
		j = i-1
		while j >= 0 and nums[j] > num:
			nums[j+1] = nums[j]
			j -= 1
		# either j == -1 or nums[j] <= num
		nums[j+1] = num

	# mergesort
	n = len(nums)
	result = [0] * n

	def merge(left, mid, right):
		nonlocal nums, result
		k = left # index_result
		i, j = left, mid + 1
		m, n = mid + 1, right + 1

		# fill from both
		while i < m and j < n:
			if nums[i] < nums[j]:
				result[k] = nums[i]
				i += 1
			else:
				result[k] = nums[j]
				j += 1
			k += 1

		while i < m:
			result[k] = nums[i]
			i += 1
			k += 1

		while j < n:
			result[k] = nums[j]
			j += 1
			k += 1

		for k in range(left, right + 1):
			nums[k] = result[k]

	def mergesort(left, right):
		if left >= right:
			return

		mid = (left + right) // 2
		mergesort(left, mid)
		mergesort(mid + 1, right)
		merge(left, mid, right)

	mergesort(0, n-1)
	
	# quicksort
	def partition(left, right, pivot):
		p_index = left # first index of the segment that's larger than pivot
		for i in range(left, right + 1):
			if nums[i] <= pivot:
				nums[p_index], nums[i] = nums[i], nums[p_index]
				p_index += 1
		return p_index

	def quicksort(left, right):
		if left >= right:
			return
		
		p_index = partition(left, right-1, nums[right])
		nums[p_index], nums[right] = nums[right], nums[p_index]
		quicksort(left, p_index - 1)
		quicksort(p_index + 1, right)
	
	quicksort(0, len(nums)-1)
	
	# heapsort in place
	size = n

	def heapify(maxheap):
		nonlocal size
		for i in range(size // 2, 0, -1):
			bubble_down(maxheap, i)

	def heappop(maxheap):
		nonlocal size
		if size == 0:
			return

		maxheap[1], maxheap[size] = maxheap[size], maxheap[1]
		popped = maxheap[size]
		size -= 1
		bubble_down(maxheap, 1)
		return popped

	def bubble_down(maxheap, index):
		nonlocal size
		if index > size:
			return

		largest_index = index
		left_index = left(index)
		right_index = right(index)

		if left_index <= size and maxheap[largest_index] < maxheap[left_index]:
			largest_index = left_index
		if right_index <= size and maxheap[largest_index] < maxheap[right_index]:
			largest_index = right_index

		if largest_index == index:
			return
		maxheap[index], maxheap[largest_index] = maxheap[largest_index], maxheap[index]
		bubble_down(maxheap, largest_index)

	def parent(index):
		return index // 2

	def left(index):
		return index * 2

	def right(index):
		return left(index) + 1

	# sorting algorithm
	maxheap = [0] + nums
	heapify(maxheap)

	for _ in range(n):
		popped = heappop(maxheap)

	return maxheap[1:]