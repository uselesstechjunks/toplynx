"""
862. Shortest Subarray with Sum at Least K
https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/
Given an integer array nums and an integer k, return the length of the shortest 
non-empty subarray of nums with a sum of at least k. 
If there is no such subarray, return -1.
A subarray is a contiguous part of an array.

Note:
This is not the optimal solution for this problem.
That uses monotonic queue and works in O(n).
This one is useful when there are updates to the array.
"""
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.nodes = [0]*4*self.n
        def build(i, l, r):
            nonlocal arr
            if l > r: return            
            if l == r:
                self.nodes[i] = arr[l]
            else:
                m = l+(r-l)//2
                build(2*i, l, m)
                build(2*i+1, m+1, r)
                self.nodes[i] = min(self.nodes[2*i], self.nodes[2*i+1])
        build(1, 0, self.n-1)
    def search(self, r, q):
        def impl(i, tl, tr, r, q):
            if tl > r: # invalid range
                return -1
            if self.nodes[i] > q:
                return -1
            if tl == tr: # leaf
                return tl
            else:
                tm = (tl+tr)//2
                if tm < r:
                    if self.nodes[2*i+1] <= q:
                        return impl(2*i+1, tm+1, tr, r, q)
                    else: # self.tree[2*i] <= q
                        return impl(2*i, tl, tm, r, q)
                else:
                    return impl(2*i, tl, tm, r, q)
        x = impl(1, 0, self.n-1, r, q)
        return x
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        if len([True for num in nums if num >= k]) > 0:
            return 1
        n = len(nums)
        # sums keeps the prefix sum
        sums = nums[:]
        for i in range(1, n):
            sums[i] += sums[i-1]
        # For each Q=sums[r], need to find rightmost l
        # such that sums[r]-sum[l] >= k
        # Idea: Find rightmost x <= sums[r]-k using segment tree.
        # O(n) build time, n queries with O(log n), overall O(n log n)
        tree = SegmentTree(sums)
        minLen = 2*n
        for r in range(1, n):
            minLen = min(minLen, r+1) if sums[r] >= k else minLen
            l = tree.search(r-1, sums[r]-k)
            minLen = min(minLen, r-l) if l != -1 else minLen
        return minLen if minLen < 2*n else -1
