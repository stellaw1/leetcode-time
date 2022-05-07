# LeetCode Cheatsheet

A compilation of notes for Leetcode problems I have worked on

## Source
https://seanprashad.com/leetcode-patterns/


## Quick Access Links 
- [LeetCode](#leetcode-cheatsheet)
    - [Source](#source)
- [Easy](#easy-problems)
    - [DP](#dp)
        - [53 - Climbing Stairs](#53---maximum-subarray)
        - [70 - Climbing Stairs](#70---climbing-stairs)
- [Medium](#medium-problems)
    - [53 - Climbing Stairs](#53---maximum-subarray)
- [Hard](#hard-problems)
    - [53 - Climbing Stairs](#53---maximum-subarray)



# Easy problems

## DP
***

# [53](https://leetcode.com/problems/maximum-subarray/) - Maximum Subarray

Find conitguous subarray with largest sum. 


## Solution:
**Iterative**
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        memo = [0] * n
        memo[0] = nums[0]
        
        for i in range(1, n):
            memo[i] = max(memo[i-1] + nums[i], nums[i])
            
        return max(memo)
```
**Kadane's algorithm**
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        curr_sum = 0

        for i in nums: 
            curr_sum = max(curr_sum + i, i)
            max_sum = max(max_sum, curr_sum)

        return max_sum
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n) for Iterative O(1) for Kadane's
```

## Notes: 
- memoize sum of max subarray in [0:i] including element i for each element
- Kadane's algorithm has better space complexity

<br><br>



# [70](https://leetcode.com/problems/climbing-stairs/) - Climbing Stairs

Number of distinct ways to take n steps using either steps of size 1 or 2. 

## Solution: 
**Iterative**
```python 
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = [0]
        memo.append(1)
        memo.append(2)
        
        for i in range(3, n + 1):
            memo.append(memo[i - 1] + memo[i - 2])
            
        return memo[n]
```
**Recursive**
```python 
class Solution:
    def climbStairs(self, n: int, memo = {}) -> int:
        if n in memo:
            return memo[n]
        elif n <= 2:
            memo[n] = n
            return memo[n]
        else:
            memo[n] = self.climbStairs(n - 1, memo) + self.climbStairs(n - 2, memo)
            return memo[n]
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```

## Notes:
- Fibonacci
- memoize 

<br><br>
***


# Medium problems




# Hard problems
