def findKthLargest(self, nums: List[int], k: int) -> int:
	def quickselect():
		# the idea is pick a number at a random index
		# partition the array to find the correct index of that number
		# array: [left_index, p_index-1][p_index][p_index+1, right]
		#                   L               P             R
		# if len(L) == k, P is answer
		# if len(L) > k, we recurse into the left subarray for k-th largest
		# if len(L < k, we recurse into the right subarray for k-len(L)-th largest
		def partition(left, right, pivot):
			p_index, index = left, left
			while index <= right:
				if nums[index] < pivot:
					nums[p_index], nums[index] = nums[index], nums[p_index]
					p_index += 1
				index += 1
			return p_index

		def quickselect_impl(left, right, k):
			if left > right:
				return None
			
			# randomly select a partiton index as pivot and move it to the right
			p_index = random.randint(left, right)
			nums[p_index], nums[right] = nums[right], nums[p_index]

			# partition the array based on pivot
			p_index = partition(left, right-1, nums[right])
			nums[p_index], nums[right] = nums[right], nums[p_index]

			size_left = p_index - left + 1
			if size_left-1 == k:
				return nums[p_index]

			if k < size_left-1:
				return quickselect_impl(left, p_index-1, k)

			return quickselect_impl(p_index+1, right, k-size_left)

		n = len(nums)
		return impl(0, n-1, n-k)
	
	def heap():
		# need to find k-th largest
		# so need a minheap so that whne something larger comes up,
		# we pop the smallest
		# the heap stores all numbers greater than or equal to top
		minheap = []
		for num in nums:
			heapq.heappush(minheap, num)
			if len(minheap) > k:
				heapq.heappop(minheap)
		return minheap[0] if len(minheap) else 0
	
	return heap()
	# return quickselect()