"""
Contains Duplicate: 
Type: Easy

Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
Example 1:

Input: nums = [1,2,3,1]
Output: true

Example 2:

Input: nums = [1,2,3,4]
Output: false

Example 3:

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
"""
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        d = {}
        for num in nums:
            if d.get(num):
                return True
            else:
                d[num] = 1
        return False


"""
Valid Anagram
Type: Easy
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true
Example 2:

Input: s = "rat", t = "car"
Output: false

"""

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        t_dict = {}
        for w in t:
            if t_dict.get(w):
                t_dict[w] += 1
            else:
                t_dict[w] = 1
        
        for st in s:
            if t_dict.get(st, 0) > 0:
                t_dict[st] -= 1
            else:
                return False
        return True
    

"""
Two Sum
Type: Easy

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
"""

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        sum_dict = {}
        for index, num in enumerate(nums):
            if sum_dict.get(num):
                return [sum_dict[num] -1 , index]
            else:
                sum_dict[target - num] = index + 1
        

"""Better One"""

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        prevMap = {}  # val -> index

        for i, n in enumerate(nums):
            diff = target - n
            if diff in prevMap:
                return [prevMap[diff], i]
            prevMap[n] = i


"""
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Type: Medium 

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Example 2:

Input: strs = [""]
Output: [[""]]
Example 3:

Input: strs = ["a"]
Output: [["a"]]
"""

from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = defaultdict(list)

        for st in strs:
            count = [0] * 26
            for c in st:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(st)
        return ans.values()