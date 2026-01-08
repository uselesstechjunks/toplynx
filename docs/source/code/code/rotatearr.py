from random import shuffle

def reverse(nums, l, r):
	while l < r:
		nums[l], nums[r] = nums[r], nums[l]
		l += 1
		r -= 1

def rotateRight(nums, k):
	n = len(nums)
	reverse(nums, 0, n-1)
	reverse(nums, 0, k-1)
	reverse(nums, k, n-1)

def rotateLeft(nums, k):
	n = len(nums)
	reverse(nums, 0, k-1)
	reverse(nums, k, n-1)
	reverse(nums, 0, n-1)

if __name__ == '__main__':
	nums = list(range(10))
	shuffle(nums)
	print(nums)
	rotateRight(nums, 3)
	print(nums)
	rotateLeft(nums, 3)
	print(nums)