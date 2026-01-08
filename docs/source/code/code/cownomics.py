from typing import List

class Solution:
    """
    Problem: https://usaco.org/index.php?page=viewproblem2&cpid=741
    """
    def min_diff(self, n: int, m: int, sequences: List[str]) -> int:
        
        def condition_satisfied(length: int) -> bool:
            for start in range(m - length):
                spotty = set([sequence[start:start + length] for sequence in sequences[:n]])
                if not any(sequence[start:start + length] in spotty for sequence in sequences[n:]):
                    return True
            return False

        left, right = 1, m
        while left < right:
            mid = (left + right) // 2
            if condition_satisfied(mid):
                right = mid
            else:
                left = mid + 1
        return left
    
if __name__ == '__main__':
    with open('cownomics.in', 'r') as input:
        n, m = map(int, input.readline().strip().split())
        sequences = [x.strip() for x in input.readlines()]
    
    solution = Solution()
    ans = solution.min_diff(n, m, sequences)
    
    with open('cownomics.out', 'w') as output:
        print(ans, file=output)