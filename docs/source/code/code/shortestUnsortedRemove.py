def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
	length = len(arr)
	left, right = 0, len(arr)-1
	while left < length-1 and arr[left] <= arr[left+1]:
		left += 1
	while right > 0 and arr[right-1] <= arr[right]:
		right -= 1
	if left >= right:
		return 0
	# print(left, right)
	min_width = min(length - left - 1, right)
	# print(min_width)
	for l in range(left + 1):
		r = right
		while r < length and arr[l] > arr[r]:
			r += 1
		min_width = min(min_width, r-l-1)
	return min_width