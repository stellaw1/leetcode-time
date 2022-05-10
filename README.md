# LeetCode Cheatsheet

A compilation of notes for Leetcode problems I have worked on

## Source
https://seanprashad.com/leetcode-patterns/


## Quick Access Links 
- [LeetCode](#leetcode-cheatsheet)
    - [Source](#source)
- [Easy](#easy-problems)
    - [Arrays](#arrays)
        - [1 - Two Sum](#1---two-sum)
        - [167 - Two Sum II](#167---two-sum-ii)
        - [15 - 3Sum](#15---3sum)
        - [217 - Contains Duplicate](#217---contains-duplicate)
        - [2022 - Convert 1D Array Into 2D Array](#2022---convert-1d-array-into-2d-array)
    - [Linked List](#linked-list)
        - [206 - Reverse Linked List](#206---reverse-linked-list)
    - [DP](#dp)
        - [53 - Maximum Subarray](#53---maximum-subarray)
        - [70 - Climbing Stairs](#70---climbing-stairs)
        - [338 - Counting Bits](#338---counting-bits)
    - [Trees](#trees)
        - [100 - Same Tree](#100---same-tree)
- [Medium](#medium-problems)
    - [53 - Climbing Stairs](#53---maximum-subarray)
- [Hard](#hard-problems)
    - [53 - Climbing Stairs](#53---maximum-subarray)


<br><br><br>
# Easy problems
<br>

## Arrays
***
<br>

# [1](https://leetcode.com/problems/two-sum/) - Two Sum

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.


## Solution:
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        record = {}
        
        for i in range(len(nums)):
            need = target - nums[i]
            if need in record:
                return [i, record[need]]
            else:
                record[nums[i]] = i
        return
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```

## Notes:
- use HashMap 


<br><br>

# [167](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) - Two Sum II

Given a 1-indexed, non-ascending array of integers and a target, return indices of the two numbers such that they add up to target.


## Solution:
```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1
        
        while j > i:
            sum = numbers[i] + numbers[j]
            if sum == target:
                return [i + 1, j + 1]
            elif sum < target:
                i += 1
            else:
                j -= 1
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```

## Notes:
- use two pointers 


<br><br>

# [15](https://leetcode.com/problems/3sum/) - 3Sum

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.


## Solution:
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ret = []
        
        for i in range(n - 2):
            # check if duplicate
            if i != 0 and nums[i] == nums[i - 1]:
                continue
            # break when positive number is reached
            elif nums[i] > 0:
                break
            # find second and third element like in two sum II
            else:
                l = i + 1
                r = n - 1
                
                while r > l:
                    sum = nums[l] + nums[r]
                    if sum == -nums[i]:
                        ret.append([nums[i], nums[l], nums[r]])
                        
                    if sum <= -nums[i]:
                        l += 1
                        while nums[l - 1] == nums[l] and l < r:
                            l += 1
                    if sum >= -nums[i]:
                        r -= 1
                        while nums[r + 1] == nums[r] and l < r:
                            r -= 1
                        
        return ret
```

## Complexity Analysis:
```
* Time complexity:   O(n^2)
* Space complexity:  O(1)
```

## Notes:
- sort array to avoid duplicate and can terminate early upon reaching positive element
- finding second and third element in triplet reduces to two sum II


<br><br>


# [217](https://leetcode.com/problems/contains-duplicate/) - Contains Duplicate

Return true if there is a duplicate value in array


## Solution:
```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        history = set()
        
        for i in range(len(nums)):
            if nums[i] in history:
                return True
            history.add(nums[i])
        return False
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```

<br><br>

# [2022](https://leetcode.com/problems/convert-1d-array-into-2d-array/) - Convert 1D Array Into 2D Array

Convert 1D array into 2D array of dimension m by n. 


## Solution:
```python
class Solution:
    def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
        if len(original) != m * n:
            return []
        else:
            mat = []
            
            j = 0
            for i in range(m):
                row = original[j:j + n]
                mat.append(row)
                j += n
                
            return mat
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```


<br><br>
## Linked List
***
<br>


# [141](https://leetcode.com/problems/linked-list-cycle/) - Linked List Cycle

Given head, the head of a linked list, determine if the linked list has a cycle in it.


## Solution:
**Brute Force**
```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head is None or head.next is None:
            return False
        
        record = set()
        record.add(head)
        curr = head
        
        while curr.next:
            if curr.next in record:
                return True
            curr = curr.next
            record.add(curr)
        return False
```
**Floyd's Cycle Finding Algorithm**
```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast = head
        slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n) or O(1) for Floyd's
```

## Notes: 
- aka Hare-Tortoise algorithm
- fast and slow pointers are bound to overlap as they repeat cycle until eventually overlapping

<br><br>

# [206](https://leetcode.com/problems/reverse-linked-list/) - Reverse Linked List

Given the head of a linked list return the reverse linked list. 


## Solution:
**Iterative**
```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is  None or head.next is None:
            return head
        
        nxt = None
        prev = head
        curr = head.next
        prev.next = None
        
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        
        return prev
```
**Recursive**
```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        prev = self.reverseList(head.next)
        head.next.next = head
        head.next = None

        return prev
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```

## Notes: 
- remember to set new tail's next to be `None`




<br><br>
## DP
***
<br>

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

<br><br>

# [338](https://leetcode.com/problems/counting-bits/) - Counting Bits

Return an array of length n + 1 such that ans[i] is the number of 1's in the binary representation of i.

## Solution: 
**Iterative**
```python 
class Solution:
    def countBits(self, n: int) -> List[int]:
        if n == 0:
            return [0]
        
        ans = [None] * (n + 1)
        ans[0] = 0
        ans[1] = 1
        counter = 0
        
        for i in range(2, n + 1):
            # check if i is a power of 2
            if (i & (i-1) == 0):
                ans[i] = 1
                counter = 1
            else:
                ans[i] = 1 + ans[counter]
                counter += 1
                
        return ans
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```

## Notes:
- pattern equals beginning of array + 1, restarting at beginning of array every power of 2
- `i & (i-1)` checks if i is a power of 2

***
<br><br>

## Trees
***
<br>

# [100](https://leetcode.com/problems/same-tree/) - Same Tree

Given 2 roots, return whether they are the same tree or not


## Solution:
**DFS**
```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        elif p and q:
            # check value and recuse on left and right children
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else: 
            return False
```
**BFS**
```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # base case
        if not p and not q:
            return True
        if not p or not q:
            return False

        pq = [p]
        qq = [q]
        
        while pq and qq: 
            currp = pq.pop()
            currq = qq.pop()
            
            # check value
            if currp.val != currq.val:
                return False

            # append children, returning false if number of children don't match
            if currp.left is not None and currq.left is not None:
                pq.append(currp.left)
                qq.append(currq.left)
            elif currp.left or currq.left:
                return False
                
            if currp.right is not None and currq.right is not None:
                pq.append(currp.right)
                qq.append(currq.right)
            elif currp.right or currq.right:
                return False
        
        return len(pq) == len(qq)
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```

<br><br>


# [543](https://leetcode.com/problems/diameter-of-binary-tree/) - Diameter of Binary Tree

Return the length of the diameter of the tree.


## Solution:
```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        
        diameter = [0]
        
        self.getHeight(root, diameter)
        
        return diameter[0] - 1
    
    # calculate height of subtree rooted at node while checking for max sum of subtree heights 
    def getHeight(self, node, diameter):
        if node is None:
            return 0
        
        l = self.getHeight(node.left, diameter)
        r = self.getHeight(node.right, diameter)
        
        diameter[0] = max(diameter[0], 1 + l + r)
        
        return 1 + max(l, r)
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```

## Notes:
- **diameter**: length of the longest path between any two nodes in a tree (may or may not pass through the root)

<br><br>


<br><br><br>
# Medium problems




# Hard problems
