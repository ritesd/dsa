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
    Depth and Height of Tree: Depth is number of edges from the root to the node.
    degree of a node: is the total number of branches of that node.
    height of a node: is the number of edges from the node to the deepest leaf
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

class BinaryTree:
    """
    Binary Tree: Binary Trees are hierarchical data structures in which each node has at most two children, often referred to as the left and right child
        Full Binary Tree: A Full Binary Tree is a binary tree in which every node has 0 or 2 children
        Perfect Binary Tree: A Perfect Binary Tree is a binary tree in which all interior nodes have 2 children and all the leaf nodes have the same depth.
        Complete Binary Tree: A Complete Binary tree is a binary tree in which every level, except possibly the last, is completely filled. All the nodes in a complete binary tree are as far left as possible.
        
    """