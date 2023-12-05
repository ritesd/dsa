"""This file for testing dsa"""
import time
from dsa import (
    CirculerLinkedList,
    Tree,
    BinaryTreeNode,
    Btree,
    BtreeArray,
    BinarySearchTree,
    TNode,
)


def test_linked_list():
    """testing linked list"""
    cl = CirculerLinkedList()
    for value in range(1, 10):
        cl.add(value=value)

    for obj in cl:
        print(obj.value)


def tree():
    tree = Tree("Drinks", [])
    cold = Tree("Cold", [])
    hot = Tree("Hot", [])
    tree.add_child(cold)
    tree.add_child(hot)
    tea = Tree("Latte", [])
    coffee = Tree("Mocha", [])
    coke = Tree("Coke", [])
    sprite = Tree("Sprite", [])
    cold.add_child(coke)
    cold.add_child(sprite)
    hot.add_child(tea)
    hot.add_child(coffee)
    print(tree)


def test_btree_to_array():
    root = BinaryTreeNode(20)
    left = BinaryTreeNode(100)
    right = BinaryTreeNode(3)
    root.leftchild = left
    root.rightchild = right
    left.leftchild = BinaryTreeNode(50)
    left.rightchild = BinaryTreeNode(15)
    right.leftchild = BinaryTreeNode(250)
    right.rightchild = BinaryTreeNode(35)
    left.leftchild.leftchild = BinaryTreeNode(222)

    def to_array(node, arr, loc):
        if not node:
            return arr
        if node.leftchild:
            arr[loc * 2] = node.leftchild.data
            arr = to_array(node.leftchild, arr, loc * 2)
        if node.rightchild:
            arr[loc * 2 + 1] = node.rightchild.data
            arr = to_array(node.rightchild, arr, loc * 2 + 1)
        return arr

    arr = [0] * 9
    arr[1] = root.data
    print(to_array(root, arr, 1))


def order_traversal():
    root = BinaryTreeNode(20)
    left = BinaryTreeNode(100)
    right = BinaryTreeNode(3)
    root.leftchild = left
    root.rightchild = right
    left.leftchild = BinaryTreeNode(50)
    left.rightchild = BinaryTreeNode(15)
    right.leftchild = BinaryTreeNode(250)
    right.rightchild = BinaryTreeNode(35)
    left.leftchild.leftchild = BinaryTreeNode(222)
    # Btree().pre_order_traversal(root=root)

    # # below is in-order
    # print("#### Inorder Traversal")

    # Btree().in_order_traversal(root=root)

    # print("\n #### Post Traversal \n")

    # Btree().post_order_traversal(root=root)

    # print("\n ## Level order traversal \n")

    # Btree().level_order_traversal(root=root)

    # print("\n ## search \n")
    import ipdb

    ipdb.set_trace()
    print(Btree().insert_node_btree(root, 90))

    # print(Btree().search_element(root, 90))

    Btree().level_order_traversal(root=root)
    time.sleep(5)

    print("#### deleting node")
    print(Btree().delete_node_btree(root=root, data=20))

    print("#### print tree")
    time.sleep(5)
    Btree().level_order_traversal(root=root)


def test_array_btree():
    root = BinaryTreeNode(20)
    left = BinaryTreeNode(100)
    right = BinaryTreeNode(3)
    root.leftchild = left
    root.rightchild = right
    left.leftchild = BinaryTreeNode(50)
    left.rightchild = BinaryTreeNode(15)
    right.leftchild = BinaryTreeNode(250)
    right.rightchild = BinaryTreeNode(35)
    left.leftchild.leftchild = BinaryTreeNode(222)

    def to_array(node, arr, loc):
        if not node:
            return arr
        if node.leftchild:
            arr[loc * 2] = node.leftchild.data
            arr = to_array(node.leftchild, arr, loc * 2)
        if node.rightchild:
            arr[loc * 2 + 1] = node.rightchild.data
            arr = to_array(node.rightchild, arr, loc * 2 + 1)
        return arr

    btree = BtreeArray(9)
    btree.insert_value(root.data)
    to_array(root, btree.array, 1)
    print(btree.array)
    btree.last_updated = 8

    btree.inorder_traversal(1)


def test_bst():
    root = TNode(70)
    left = TNode(50)
    right = TNode(90)
    root.left_child = left
    root.right_child = right
    print(f"****{root}")
    binary_search_tree = BinarySearchTree()
    binary_search_tree.bst_insert(root, 30)
    binary_search_tree.bst_insert(root, 60)
    binary_search_tree.bst_insert(root, 80)
    binary_search_tree.bst_insert(root, 100)
    binary_search_tree.bst_inorder_traversal(root)
    binary_search_tree.bst_delete_node(root, 80)
    binary_search_tree.bst_inorder_traversal(root)


if __name__ == "__main__":
    # test_linked_list()
    # tree()
    # test_btree_to_array()
    # order_traversal()
    # test_array_btree()
    test_bst()
