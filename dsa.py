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
        queue = []
        queue.append(root)

        while len(queue):
            # enqueue child of 1st element
            queue.append(queue[0].leftchild) if queue[0].leftchild else None
            queue.append(queue[0].rightchild) if queue[0].rightchild else None
            print(queue.pop(0).data)
    
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
