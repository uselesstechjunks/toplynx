from typing import List, Tuple

class Solution:
    def max_dist(self, n: int, m: int, intervals: List[Tuple[int]]) -> int:
        
        def condition_satisfied(dist: int) -> bool:
            num_placed, last_placed = 0, intervals[0][0] - dist
            for first, last in intervals:
                if last_placed + dist < first:
                    # this is important so that the next cow is placed at the beginning of the current interval
                    last_placed = first - dist
                while num_placed < n and first <= last_placed + dist <= last:
                    num_placed += 1
                    last_placed += dist
                if num_placed == n:
                    break
            return num_placed == n
        
        intervals.sort()
        # binary search on whether it's valid or not to place cows at a specific distance
        left, right = 0, intervals[-1][1] - intervals[0][0]
        while left < right:
            mid = (left +  right + 1) // 2
            if condition_satisfied(mid):
                left = mid
            else:
                right = mid - 1
        return left

if __name__ == '__main__':
    with open('socdist.in', 'r') as input:
        n, m = map(int, input.readline().strip().split())
        intervals = [tuple(int(x) for x in line.strip().split()) for line in input.readlines()]
    
    assert(m == len(intervals))
    solution = Solution()
    ans = solution.max_dist(n, m, intervals)
    
    with open('socdist.out', 'w') as output:
        print(ans, file=output)