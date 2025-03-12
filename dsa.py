"""
This file contain the linear and non-linear data structure 
"""

class Node:
    """
    Node to store the value and reference of next node.
    """
    def __init__(self, value):
        self.value = value
        self.next = None


class CirculerLinkedList:
    """
    linear type data structur cireculer linked list
    """
    def __init__(self):
        """
        Constructor for initializing linked list
        """
        self.head = None
        self.tail = None

    def create(self, value):
        """
        method for creating linked list.
        """
        node = Node(value)
        self.head = node
        self.tail = node

    def add(self, value):
        """
        add node at begining.
        """
        
        if not self.head:
            self.create(value)
        else:
            node = Node(value)
            node.next = self.head
            self.head = node
            self.tail.next = self.head

    def __iter__(self):
        temp = self.head
        while temp:
            yield temp
            temp = temp.next
            if self.head == temp:
                break

    def add_mid(self, mid_node, value):
        """ Adding node in between the linked list"""
        if mid_node is None:
            print("node does not exist")
            return
        new_node = Node(value)
        new_node.next = mid_node.next
        mid_node.next = new_node

class Tree:
    """
    Depth and Height of Tree: Depth is number of edges from the root to the node. --> 
    degree of a node: is the total number of branches of that node. --> this calculated from the same node
    height of a node: is the number of edges from the node to the deepest leaf --> this calculated from depest node
    """
    def __init__(self, data, children=[]):
        """Constructor to creat tree"""
        self.data = data
        self.children = children

    def __str__(self, level=0):
        ret = " " * level + str(self.data) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


    def add_child(self, tree_node):
        """
        method to add tree into as children
        """
        self.children.append(tree_node)

class BinaryTreeNode:
    """
    Binary Tree: Binary Trees are hierarchical data structures in which each node has at most two children, often referred to as the left and right child
        Full Binary Tree: A Full Binary Tree is a binary tree in which every node has 0 or 2 children
        Perfect Binary Tree: A Perfect Binary Tree is a binary tree in which all interior nodes have 2 children and all the leaf nodes have the same depth.
        Complete Binary Tree: A Complete Binary tree is a binary tree in which every level, except possibly the last, is completely filled. All the nodes in a complete binary tree are as far left as possible.
        
    """
    def __init__(self, data) -> None:
        self.data = data
        self.leftchild = None
        self.rightchild = None


class Btree:

    # def __init__(self, root: BinaryTreeNode) -> None:
    #     self.root = root

    def pre_order_traversal(cls, root: BinaryTreeNode):

        if not root:
            return
        print(root.data)

        cls.pre_order_traversal(root=root.leftchild)
        cls.pre_order_traversal(root=root.rightchild)

    def in_order_traversal(cls, root: BinaryTreeNode):
        if not root:
            return
        cls.in_order_traversal(root=root.leftchild)
        print(root.data)
        cls.in_order_traversal(root=root.rightchild)

    def post_order_traversal(cls, root: BinaryTreeNode):
        if not root:
            return
        cls.post_order_traversal(root=root.leftchild)
        cls.post_order_traversal(root=root.rightchild)
        print(root.data)

    def level_order_traversal(cls, root: BinaryTreeNode):
        # since we dont have queue data structur ready so we will use list as queue
        queue = [root]
        # queue.append(root)

        while queue:
            # enqueue child of 1st element
            node = queue.pop(0)
            print(node.data)
            if node.leftchild:
                queue.append(node.leftchild)
            if node.rightchild:
                queue.append(node.rightchild)
    
    def search_element(cls, root, find_data):
        """
        we will implement queue as it is faster than stack,
        Now to use queue we will use level order T
        """
        if not root:
            return "empty root"
        queue = []
        queue.append(root)

        while len(queue):
            # enqueue child of 1st element
            node = queue.pop(0)
            if node.data == find_data:
                return "Found"
            queue.append(node.leftchild) if node.leftchild else None
            queue.append(node.rightchild) if node.rightchild else None
        return "Not found"
    
    def insert_node_btree(cls, root: BinaryTreeNode, data):
        if root is None:
            root = BinaryTreeNode(data=data)
            return "Inserted successfuly"
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.leftchild:
                queue.append(node.leftchild)
            else:
                node.leftchild = BinaryTreeNode(data=data)
                return "Inserted successfuly"
            if node.rightchild:
                queue.append(node.rightchild)
            else:
                node.rightchild = BinaryTreeNode(data=data)
                return "Inserted Successfuly"
    
    def delete_node_btree(cls, root: BinaryTreeNode, data: int):
        if root is None:
            return "Tree is empty"
        
        node_to_be_delete = None
        queue = [root]

        while queue:
            node = queue.pop(0)
            if not node_to_be_delete and node.data == data:
                node_to_be_delete = node
                """did not break the loop in order to find the depest node"""
            if node.leftchild:
                queue.append(node.leftchild)
            if node.rightchild:
                queue.append(node.rightchild)
        
        if node_to_be_delete:
            node_to_be_delete.data = node.data
            cls.delete_deepest_node(root, node)
            return "Node found and deleted"
        return "Node not found"

    def delete_deepest_node(cls, root: BinaryTreeNode, dnode):
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node == dnode:
                node = None
            else:
                if node.leftchild:
                    if node.leftchild == dnode:
                        node.leftchild = None
                        return
                    queue.append(node.leftchild)
                if node.rightchild:
                    if node.rightchild == dnode:
                        node.rightchild = None
                        return
                    queue.append(node.rightchild)
    
class BtreeArray:
    def __init__(self, n) -> None:
        self.array = [None] * n
        self.last_updated = 0
    
    def insert_value(self, value):
        if len(self.array) == self.last_updated - 1:
            return "Array is full"
        self.array[self.last_updated + 1] = value
        self.last_updated += 1
        return "Inserted"
    
    def inorder_traversal(self, index):
        if index > self.last_updated:
            return
        
        self.inorder_traversal(index=index*2)
        print(self.array[index])
        self.inorder_traversal(index=index*2 + 1)

    def delete_node(self, value):
        for index, v in enumerate(self.array):
            if v==value:
                self.array[index] = self.array[self.last_updated]
                self.last_updated -= 1
                return "Node has been deleted"
        return "Node not found"

"""
Binany search Tree
: - I am implementing using dataclasses for node to do more awesome stuff
"""

from dataclasses import dataclass
import dataclasses
from typing import Literal, TypeVar

TNode = TypeVar('TNode')

@dataclass(order=True)
class TNode:
    data: int
    left_child: TNode = None
    right_child: TNode = None

    def to_dict(self):
        return dataclasses.asdict(self)
    
@dataclass(order=True)
class AVLTNode:
    data: int
    left_child: TNode = None
    right_child: TNode = None
    height: int = None

    def to_dict(self):
        return dataclasses.asdict(self)

class BinarySearchTree:

    def bst_search(cls, root: TNode, value)-> TNode:
        if not root:
            return None

        if root.data == value:
            return root
        if value < root.data:
            return cls.bst_search(root=root.left_child, value=value)
        else:
            print(root.right_child)
            return cls.bst_search(root=root.right_child, value=value)
        
    def bst_insert(cls, current_node: TNode, value):
        if current_node is None:
            current_node = TNode(value)

        elif value <= current_node.data:
            current_node.left_child = cls.bst_insert(current_node.left_child, value)
        else:
            current_node.right_child = cls.bst_insert(current_node.right_child, value)
        return current_node
    
    def bst_inorder_traversal(cls, root:TNode):
        if root is None:
            return None
        cls.bst_inorder_traversal(root.left_child)
        print(root.data)
        cls.bst_inorder_traversal(root.right_child)

    def bst_search_parent_node(cls, root: TNode, node:TNode)-> TNode:
        if not root:
            return None

        if root.left_child == node or root.right_child == node:
            return root
        if node.data < root.data:
            return cls.bst_search_parent_node(root=root.left_child, node=node)
        else:
            return cls.bst_search_parent_node(root=root.right_child, node=node)

    def bst_delete_node(cls, root:TNode, delete_value):
        """
        1st: node tobe deleted is leaf node
            - then we simply set it None
        2nd: node to be deleted is having one child
            - then we simply link the parent node with child node of node to be deleted
        3rd: node to be deleted is having two child
            - then we look for lowest value successor in right subtree
            - and replace it with node to be deleted and set succesor node to None
        """

        if root is None:
            return "Tree is empty"
        if delete_value < root.data:
            root.left_child = cls.bst_delete_node(root.left_child, delete_value)
        elif delete_value > root.data:
            root.right_child = cls.bst_delete_node(root.right_child, delete_value)
        else:
            if root.left_child is None:
                temp = root.right_child
                root = None
                return temp
            elif root.right_child is None:
                temp = root.left_child
                root = None
                return temp
            temp = cls._get_successor(root.right_child)
            root.data = temp.data
            root.right_child = cls.bst_delete_node(root.right_child, temp.data)
        return root

    def _get_successor(cls, root: TNode)-> TNode:
        """also called as getMinvalueNode"""
        if root is None or root.left_child is None:
            return root
        else:
            return cls._get_successor(root.left_child)


class AVLTree(BinarySearchTree):
    """
    AVL Tree is a self-balancing Binary Search Tree
    """
    # def __init__(self, data):
    #     super().__init__(data)

    def insert(cls, root: AVLTNode, data):
        root = cls.bst_insert_node(root, data)

    def delete(cls, root, delete_value):
        root = cls.bst_delete_node(root, delete_value)

    def calculate_height(cls, root: AVLTNode):
        """
        Calculate height of the tree
        """
        if root is None:
            return 0
        else:
            return max(cls.calculate_height(root.left_child), cls.calculate_height(root.right_child)) + 1
    
    def right_rotate(cls, root: AVLTNode):
        """
        Right rotation
        time complexity - O(1)
        space complexity - O(1)
        """
        new_root = root.left_child
        root.left_child = root.left_child.right_child
        new_root.right_child = root
        root.height = cls.calculate_height(root=root)
        new_root.height = cls.calculate_height(root=new_root)
        return new_root

    def left_rotate(cls, root: AVLTNode):
        """
        Left rotation
        """
        new_root = root.right_child
        root.right_child = new_root.right_child.left_child
        new_root.left_child = root
        root.height = cls.calculate_height(root=root)
        new_root.height = cls.calculate_height(root=new_root)
        return new_root
        
class BHeap:
    MAX = "max"
    MIN = "min"
    def __init__(self, size, type=Literal[MAX, MIN]):
        if type in [self.MAX, self.MIN]:
            self._type = type
        self._customeList = [None] * (size + 1)
        self._heap_size = 0
        self.maxSize = size + 1


    def peak(self):
        if self._heap_size:
            return self._customeList[1]
    
    def __sizeof__(self):
        return self._heap_size
    
    def level_order_traversal(self):
        if not self._heap_size:
            return
        for i in range(1, self._heap_size + 1):
            print(self._customeList[i])
    
    def insert(self, value):
        if self._heap_size + 1 == self.maxSize:
            return "Binary Heap is Fully Occupied"
        self._heap_size += 1
        self._customeList[self._heap_size] = value
        self.__heapify_bottom_to_top_insert(self._heap_size)
    
    def __heapify_bottom_to_top(self, index: int):
        parent_index = int(index/2)
        if parent_index <= 1:
            return
        if self._type == self.MIN:
            if self._customeList[index] < self._customeList[parent_index]:
                temp = self._customeList[parent_index]
                self._customeList[parent_index] = self._customeList[index]
                self._customeList[index] = temp
            self.__heapify_bottom_to_top(parent_index)

        if self._type == self.MAX:
            if self._customeList[index] > self._customeList[parent_index]:
                temp = self._customeList[parent_index]
                self._customeList[parent_index] = self._customeList[index]
                self._customeList[index] = temp
            self.__heapify_bottom_to_top(parent_index)
    
    def __heapify_top_to_bottom(self, index: int):
        _left_child = index * 2
        _right_child = index * 2 + 1
        _swap_child = 0
        if self._heap_size < _left_child:
            return
        if self._heap_size == _left_child:
            if self._type == self.MIN:
                if self._customeList[index] > self._customeList[_left_child]:
                    temp = self._customeList[index]
                    self._customeList[index] = self._customeList[_left_child]
                    self._customeList[_left_child] = temp
                return
            else:
                if self._customeList[index] < self._customeList[_left_child]:
                    temp = self._customeList[index]
                    self._customeList[index] = self._customeList[_left_child]
                    self._customeList[_left_child] = temp
                return
        else:
            if self._type == self.MIN:
                if self._customeList[_left_child] < self._customeList[_right_child]:
                    _swap_child = _left_child
                else:
                    _swap_child = _right_child
                if self._customeList[index] > self._customeList[_swap_child]:
                    temp = self._customeList[index]
                    self._customeList[index] = self._customeList[_swap_child]
                    self._customeList[_swap_child] = temp
            else:
                if self._customeList[_left_child] > self._customeList[_right_child]:
                    _swap_child = _left_child
                else:
                    _swap_child = _right_child
                if self._customeList[index] < self._customeList[_swap_child]:
                    temp = self._customeList[index]
                    self._customeList[index] = self._customeList[_swap_child]
                    self._customeList[_swap_child] = temp
        self.__heapify_top_to_bottom(_swap_child)
    
    def extract(self):
        if self._heap_size == 0:
            return "Binary Heap is Empty"
        extracted = self._customeList[1]
        self._customeList[1] = self._customeList[self._heap_size]
        self._customeList[self._heap_size] = None
        self.__heapify_top_to_bottom(1)
        return extracted


class TrieNode:

    def __init__(self):
        self.hashmap = {}
        self.is_endof_word = False
    
    # @property
    # def hashmap(self, key):
    #     return self.__hashmap.get(key)
    
    # @hashmap.setter
    # def hashmap(self, key, value):
    #     self.__hashmap[key] = value

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        current = self.root
        # iterate through word
        for ch in word:
            node = current.hashmap.get(ch)
            if node == None:
                node = TrieNode()
                current.hashmap.update({ch: node})
            current = node
        current.is_endof_word = True

    def search(self, word: str) -> bool:
        current = self.root
        for ch in word:
            node = current.hashmap.get(ch)
            if node == None:
                print("word not found")
                return False
            current = node
        if current.is_endof_word == True:
            print("Word found")
        else:
            print("Word not found")
        return current.is_endof_word

    def deleteString(self, word, index, root: TrieNode = None):
        # Case 1: string is already prefix of other word
        # Case 2: this word is prefix of other word
        # Case 3: Some other word is prefix of this word
        # Case 4: No is dependent of this word

        ch = word[index]
        root = self.root if not root else root
        current_node: TrieNode = root.hashmap.get("ch")
        canNodeBeDeleted = False

        # Case 1: string is already prefix of other word
        if len(current_node.hashmap) > 1:
            self.deleteString(word=word, index=index+1, root=current_node)
            return False

        # Case 2: this word is prefix of other word
        if index == len(word) - 1:
            if len(current_node.hashmap) >= 1:
                current_node.is_endof_word = False
                return False
            else:
                root.hashmap.pop(ch)
                return True

        # Case 3: Some other word is prefix of this word
        if current_node.is_endof_word:
            self.deleteString(word=word, index=index+1, root=current_node)
            return False
        
        canNodeBeDeleted = self.deleteString(word=word, index=index + 1, root=current_node)
        # Case 4: No is dependent of this word
        if canNodeBeDeleted:
            root.hashmap.pop('ch')
            return True
        else:
            return False

class Hashing:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        hash_key = self._hash_function(key)
        key_exists = False
        bucket = self.table[hash_key]
        for i, kv in enumerate(bucket):
            k, v = kv
            if key == k:
                key_exists = True
                break
        if key_exists:
            bucket[i] = (key, value)
        else:
            bucket.append((key, value))

    def search(self, key):
        hash_key = self._hash_function(key)
        bucket = self.table[hash_key]
        for k, v in bucket:
            if k == key:
                return v
        return None

    def delete(self, key):
        hash_key = self._hash_function(key)
        bucket = self.table[hash_key]
        for i, kv in enumerate(bucket):
            k, v = kv
            if key == k:
                del bucket[i]
                return True
        return False
