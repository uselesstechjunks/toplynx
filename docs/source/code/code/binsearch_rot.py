def direct_search(self, nums: List[int], target: int) -> int:
	left, right = 0, len(nums) - 1
	while left <= right:
		mid = (left + right) // 2
		# mid cuts the array in 3 parts
		# nums[left...mid-1], nums[mid], nums[mid+1...right]
		# case 1: we can check the shortest part first
		if nums[mid] == target:
			return mid
		"""
		# case 2: of the two subarrays, at most one of them is rotated
		# so at least one of them is sorted
		"""
		# NOTE: it's also possible that both of them are sorted (if mid was the min)
		# we check if left part is the sorted one
		if nums[left] <= nums[mid]: # FOR DUPLICATES: check for nums[left] < nums[mid]
			""" left subarray is sorted """
			# NOTE: the <= sign instead of < (even though numbers are distinct)
			# this is to tackle cases of arrays of size 1 and 2
			if nums[left] <= target and target < nums[mid]:
				# we can check if target is contained in this sorted half
				right = mid - 1
			else:
				# target is not in the sorted half
				left = mid + 1
		else: # FOR DUPLICATES: check for elif nums[left] > nums[mid]:
			""" right subarray is sorted """
			if nums[mid] < target and target <= nums[right]:
				# we can check if target is contained in this sorted half
				left = mid + 1
			else:
				# target is not in the sorted half
				right = mid - 1
		# FOR DUPLICATES, WE NEED TO RESORT TO LINEAR SEARCH
		# else: 
		#     left += 1
	return -1

def indirect_search():
	def find_pivot():
		left, right = 0, len(nums) - 1
		while left != right and nums[left] > nums[right]:
			mid = (left + right) // 2
			""" also works: if nums[left] > nums[mid]: """
			if nums[mid] < nums[right]:
				right = mid
			else:
				left = mid + 1
		return left

	def search_with_pivot(pivot_index):
		n = len(nums)
		left, right = 0, n - 1
		while left <= right:
			mid = (left + right) // 2
			pivot = (mid + pivot_index) % n
			if target == nums[pivot]:
				return pivot
			if target < nums[pivot]:
				right = mid - 1
			else:
				left = mid + 1
		return -1

	return search_with_pivot(find_pivot())
