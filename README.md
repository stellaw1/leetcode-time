# LeetCode Cheatsheet

A compilation of notes for Leetcode problems I have worked on

## Source
https://seanprashad.com/leetcode-patterns/ <br>
https://neetcode.io/


## Quick Access Links 
- [LeetCode](#leetcode-cheatsheet)
    - [Source](#source)
- [Easy](#easy-problems)
    - [Arrays](#arrays)
        - [1 - Two Sum](#1---two-sum)
        - [136 - Single Number](#136---single-number)
        - [217 - Contains Duplicate](#217---contains-duplicate)
        - [268 - Missing Number](#268---missing-number)
        - [448 - Find All Numbers Disappeared in an Array](#448---find-all-numbers-disappeared-in-an-array)
        - [2022 - Convert 1D Array Into 2D Array](#2022---convert-1d-array-into-2d-array)
    - [Linked List](#linked-list)
        - [21 - Merge Two Sorted Lists](#21---merge-two-sorted-lists)
        - [141 - Linked List Cycle](#141---linked-list-cycle)
        - [206 - Reverse Linked List](#206---reverse-linked-list)
    - [DP](#dp)
        - [53 - Maximum Subarray](#53---maximum-subarray)
        - [70 - Climbing Stairs](#70---climbing-stairs)
        - [121 - Best Time to Buy and Sell Stock](#121---best-time-to-buy-and-sell-stock)
        - [338 - Counting Bits](#338---counting-bits)
    - [Trees](#trees)
        - [100 - Same Tree](#100---same-tree)
        - [543 - Diameter of Binary Tree](#543---diameter-of-binary-tree)
    - [Heap and PQs](#heap-and-pqs)
        - [703 - Kth Largest Element in a Stream](#703---kth-largest-element-in-a-stream)
- [Medium](#medium-problems)
    - [Arrays](#arrays)
        - [15 - 3Sum](#15---3sum)
        - [48 - Rotate Image](#48---rotate-image)
        - [49 - Group Anagrams](#49---group-anagrams)
        - [78 - Subsets](#78---subsets)
        - [167 - Two Sum II](#167---two-sum-ii)
    - [Strings](#strings)
        - [17 - Letter Combinations of a Phone Number](#17---letter-combinations-of-a-phone-number)
    - [Linked Lists](#linked-lists)
        - [143 - Reorder List](#143---reorder-list)
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

# [136](https://leetcode.com/problems/single-number/) - Single Number

Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.


## Solution:
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ret = 0
        
        for n in nums:
            ret = ret ^ n
            
        return ret
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```


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


# [268](https://leetcode.com/problems/missing-number/) - Missing Number

Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.


## Solution:
```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        tot = len(nums)
        
        for i, n in enumerate(nums):
            tot += i - n
        
        return tot
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```

<br><br>


# [448](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/) - Find All Numbers Disappeared in an Array

Given an array nums of n integers where nums[i] is in the range [1, n], return an array of all the integers in the range [1, n] that do not appear in nums.


## Solution:
```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        ret = set()
        
        for i in range(1, len(nums) + 1):
            ret.add(i)
        
        for n in nums:
            if n in ret:
                ret.remove(n)
        
        return list(ret)
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```

## Notes:
- space complexity assumes returned list does not count as extra space

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


# [21](https://leetcode.com/problems/merge-two-sorted-lists/) - Merge Two Sorted Lists

Given the heads, merge two sorted linked lists


## Solution:
**Brute Force**
```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        
        l1 = list1
        l2 = list2
        prev, head = None, None
        
        if l1.val < l2.val:
            head = l1
            prev = l1
            l1 = l1.next
        else:
            head = l2
            prev = l2
            l2 = l2.next
        
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                prev = l1
                l1 = l1.next
            else:
                prev.next = l2
                prev = l2
                l2 = l2.next
        
        if l1:
            prev.next = l1
        
        if l2: 
            prev.next = l2
            
        return head
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```


<br><br>


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



# [121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) - Best Time to Buy and Sell Stock

Given an array of stock prices on different days, return the max profit from buying on a single day and selling in a future day. 


## Solution:
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minPrice = prices[0]
        maxProfit = 0
        
        for i in range(1, len(prices)):
            maxProfit = max(maxProfit, prices[i] - minPrice)
            minPrice = min(minPrice, prices[i])
        
        return maxProfit
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n) for Iterative O(1) for Kadane's
```

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


***
<br><br>

## Heap and PQs
***
<br>

# [703](https://leetcode.com/problems/kth-largest-element-in-a-stream/submissions/) - Kth Largest Element in a Stream

Find the kth largest element in a stream


## Solution:
```java
class KthLargest {
    
    PriorityQueue<Integer> pq;
    int k;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        this.pq = new PriorityQueue<>();
        for (int n : nums) {
            this.pq.add(n);
        }
        
        // only keep k largest elements in pq
        while (this.pq.size() > this.k) {
            this.pq.poll();
        }
    }
    
    public int add(int val) {
        this.pq.add(val);
        
        // only keep k largest elements in pq
        while (this.pq.size() > this.k) {
            this.pq.poll();
        }
        
        return this.pq.peek();
    }
}
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```

<br><br>

# [1046](https://leetcode.com/problems/last-stone-weight/) - Last Stone Weight

You are given an array of integers stones where stones[i] is the weight of the ith stone.

We are playing a game with the stones. On each turn, we choose the heaviest two stones and smash them together. Suppose the heaviest two stones have weights x and y with x <= y. The result of this smash is:

If x == y, both stones are destroyed, and
If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
At the end of the game, there is at most one stone left.

Return the weight of the last remaining stone. If there are no stones left, return 0.


## Solution:
```java
class Solution {
    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        
        for (int stone : stones) {
            pq.add(stone);
        }
        
        while (pq.size() > 1) {
            int x = pq.poll();
            int y = pq.poll();
            
            if (x < y) {
                pq.add(y - x);
            } else if (x > y) {
                pq.add(x - y);
            }
        }
        
        if (pq.size() == 1) {
            return pq.peek();
        } else {
            return 0;
        }
    }
}
```


## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```

<br><br><br>
# Medium problems

## Arrays
***
<br>


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

# [48](https://leetcode.com/problems/rotate-image/) - Rotate Image

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.


## Solution:
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        
        n = len(matrix)
        
        for s in range(floor(n/2)):
            offset = 0
            for c in range(n - 1 - 2 * s):
                save = matrix[s][s + offset]

                # TL
                matrix[s][s + offset] = matrix[n - s - 1 - offset][s]
                # BL
                matrix[n - s - 1 - offset][s] = matrix[n - s - 1][n - s - 1 - offset]
                # BR
                matrix[n - s - 1][n - s - 1 - offset] = matrix[s + offset][n - s - 1]
                # TR
                matrix[s + offset][n - s - 1] = save

                offset += 1
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```

## Notes:
- `A[:] = zip(*A[::-1])`


<br><br>

# [49](https://leetcode.com/problems/group-anagrams/) - Group Anagrams

Given an array of strings strs, group the anagrams together. 


## Solution:
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hist = {}
        
        for word in strs:
            w = ''.join(sorted(word))
            if w in hist:
                hist[w].append(word)
            else:
                hist[w] = [word]
        
        ret = []
        for w in hist:
            ret.append(hist[w])
            
        return ret
```

## Complexity Analysis:
```
* Time complexity:   O(kn) with k = time to sort length of longest string
* Space complexity:  O(n)
```


<br><br>

# [78](https://leetcode.com/problems/subsets/) - Subsets

Given an integer array nums of unique elements, return all possible subsets (the power set).


## Solution:
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ps = [[]]
        
        for n in nums:
            cps = ps[:]
            for c in cps: 
                ps.append(c[:] + [n])
        
        return ps
```
### Backtracking: 
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # TODO
```

## Complexity Analysis:
```
* Time complexity:   O(n * 2^n)
* Space complexity:  O(n * 2^n) or O(n)
```



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


## Strings
***
<br>

# [17](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) - Letter Combinations of a Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.


## Solution:
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        mapping = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        
        ret = []
        
        def helper(i, s):
            if len(digits) == len(s):
                ret.append(s)
                
            else: 
                curr = mapping.get(digits[i])
                for c in curr:
                    helper(i + 1, s + c)
        
        if digits:
            helper(0, "")
            
        return ret
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(1)
```


<br><br>

## Linked Lists
***
<br>

# [143](https://leetcode.com/problems/reorder-list/submissions/) - Reorder List

Alternatingly reverse the linked list nodes. 


## Solution:
```java
class Solution {
    public void reorderList(ListNode head) {
        Stack<ListNode> myStack = new Stack<ListNode>();
        
        ListNode curr = head;
        int count = 0;
        while(curr != null) {
            count++;
            myStack.push(curr);
            curr = curr.next;
        }
        
        curr = head;
        ListNode next = new ListNode();
        
        while (count > 1) {
            count -= 2;
            
            next = curr.next;
            curr.next = myStack.pop();
            curr.next.next = next;
            curr = next;
        }
        
        if (count == 1) {
            curr.next = null;
        } else {
            curr.next.next = null;
        }
    }
}
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```


<br><br>


## Trees
***
<br>

# [105](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) - Construct Binary Tree from Preorder and Inorder Traversal

Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.


## Solution:
```java
class Solution {
    Map<Integer, Integer> rootIndex;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        rootIndex = new HashMap<Integer, Integer>();
        
        for (int i = 0; i < inorder.length; i++) {
            rootIndex.put(inorder[i], i);
        }
        
        return helper(preorder, inorder);
    }
    
    private TreeNode helper(int[] preorder, int[] inorder) {
        if (preorder.length <= 0) {
            return null;
        }
        
        TreeNode curr = new TreeNode(preorder[0]);
        
        int ind = rootIndex.get(preorder[0]);
        
        int[] leftIn = Arrays.copyOfRange(inorder, 0, ind);
        int[] rightIn = Arrays.copyOfRange(inorder, ind + 1, inorder.length);

        int[] leftPre = Arrays.copyOfRange(preorder, 1, ind + 1);
        int[] rightPre = Arrays.copyOfRange(preorder, ind + 1, preorder.length);

        curr.left = buildTree(leftPre, leftIn);
        curr.right = buildTree(rightPre, rightIn);
        
        return curr;
    }
}
```

## Complexity Analysis:
```
* Time complexity:   O(n)
* Space complexity:  O(n)
```

## Notes
- Build a hashmap to record the relation of value -> index for inorder, so that we can find the position of root in constant time.


<br><br>


# Hard problems
