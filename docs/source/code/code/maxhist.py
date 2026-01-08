class TreeNode:
    def __init__(self, idx):
        self.idx = idx
        self.left = None
        self.right = None

class CartesianTree:
    def __init__(self, nums):
        def construct(nums):
            stack = []
            for i in range(len(nums)):
                last = None
                while stack and nums[stack[-1].idx] > nums[i]:
                    last = stack.pop()
                curr = TreeNode(idx=i)
                curr.left = last
                if stack:
                    stack[-1].right = curr
                stack.append(curr)
            return stack[0]
        self.root = construct(nums)

class Solution:
    def bruteforce(self, nums):
        n, res = len(nums), 0
        for i in range(n):
            curr = inf
            for j in range(i,n):
                curr = min(curr, nums[j])
                res = max(res, curr * (j-i+1))
        return res
    def divideAndConquerNaive(self, nums):
        def findMin(l, r):
            """ can optimize this """
            # linear search -> lg(n) search with segment tree -> O(1) search with cartesian tree
            nonlocal nums
            currIdx = r
            for i in range(l,r):
                currIdx = i if nums[currIdx] > nums[i] else currIdx
            return currIdx
        def impl(l, r):
            nonlocal nums, res
            if l > r: return -1
            minIdx = findMin(l, r)
            curr = 0
            if minIdx != -1:
                left = impl(l, minIdx-1)
                right = impl(minIdx+1, r)
                curr = max(left, right, nums[minIdx] * (r-l+1))
                res = max(res, curr)
            return curr
        res = 0
        return impl(0, len(nums)-1)
    def divideAndConquerCartesianTree(self, heights):
        root = CartesianTree(heights).root
        def impl(root, l, r):
            nonlocal heights
            if root is None:
                return 0
            minHeight = heights[root.idx]
            currArea = (r-l+1) * minHeight
            leftMaxArea = impl(root.left, l, root.idx-1)
            rightMaxArea = impl(root.right, root.idx+1, r)
            return max(currArea, leftMaxArea, rightMaxArea)
        return impl(root, 0, len(heights)-1)
    def monotonicStack(self, nums):
        """ simulates cartesian tree """
        # every time something is popped, it's the min of some range
        # those ranges cover all possible ranges exhaustively
        n, res = len(nums), 0
        stack = [-1]
        for i in range(n):
            while stack[-1] >= 0 and nums[stack[-1]] > nums[i]:
                curr = nums[stack.pop()]
                # popped in the range min of curr top+1 and i-1
                l, r = stack[-1]+1, i-1
                res = max(res, curr * (r-l+1))
            stack.append(i)
        while stack[-1] >= 0:
            curr = nums[stack.pop()]
            # popped is the range min of top+1 and n-1
            l, r = stack[-1]+1, n-1
            res = max(res, curr * (r-l+1))
        return res
    def largestRectangleArea(self, heights: List[int]) -> int:
        return self.monotonicStack(heights)
