def findMin(self, nums: List[int]) -> int:
	"""
	observations: 
	1. base case: num[left] < num[right] -> left is the answer
	2. at any given time, the range [left, right] contains the answer
	3. to satisfy left should never cross right, the condition should allow left == right
	4. in the end, the range reduces to 1 and left is the answer
	"""
	def check_with_right():
		# search range: [0, n-1]
		left, right = 0, len(nums) - 1
		# loop ends when left == right or num[left] < num[right]
		while left != right and nums[left] > nums[right]:
			# for the following, num[left] > num[right]
			# find mid
			mid = (left + right) // 2
			# if num[mid] < num[right], answer lies in [left, mid]
			if nums[mid] < nums[right]:
				right = mid
			# if num[mid] > num[right], answer lies in [mid + 1, right]
			else:
				left = mid + 1
		# left contains the min
		return nums[left]
		
	def check_with_left():
		# search range: [0, n-1]
		left, right = 0, len(nums) - 1
		# loop ends when left == right or num[left] < num[right]
		while left != right and nums[left] > nums[right]:
			# for the following, num[left] > num[right]
			# find mid
			mid = (left + right) // 2
			# if num[left] > num[mid], answer lies in [left, mid]
			if nums[left] > nums[mid]:
				right = mid
			# if num[left] < num[mid], answer lies in [mid + 1, right]
			else:
				left = mid + 1
		# left contains the min
		return nums[left]