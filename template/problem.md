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