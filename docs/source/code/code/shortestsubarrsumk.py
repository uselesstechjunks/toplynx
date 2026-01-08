from itertools import accumulate
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        cumsum = [x for x in accumulate(nums)]
        print(cumsum)
        # for each r, find the largest l in [0,r-1] such that
        # cumsum[r]-cumsum[l] >= k, or cumsum[l] <= cumsum[r]-k
        minLen = 2*n
        minQ = deque()
        for r in range(n):
            minLen = min(minLen, r+1) if cumsum[r] >= k else minLen
            while minQ and cumsum[minQ[0]] <= cumsum[r]-k:
                minLen = min(minLen, r-minQ[0])
                minQ.popleft()
            while minQ and cumsum[minQ[-1]] > cumsum[r]:
                minQ.pop()
            minQ.append(r)
        return minLen if minLen < 2*n else -1