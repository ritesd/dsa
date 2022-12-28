"""This file for testing dsa"""
from dsa import CirculerLinkedList, Tree

def test_linked_list():
    """ testing linked list"""
    cl = CirculerLinkedList()
    for value in range(1,10):
        # print(value)
        cl.add(value=value)
    
    for obj in cl:
        print(obj.value)

def tree():
    tree = Tree('Drinks', [])
    cold = Tree('Cold', [])
    hot = Tree('Hot', [])
    tree.add_child(cold)
    tree.add_child(hot)
    tea = Tree('Latte', [])
    coffee = Tree('Mocha', [])
    coke = Tree('Coke', [])
    sprite = Tree('Sprite', [])
    cold.add_child(coke)
    cold.add_child(sprite)
    hot.add_child(tea)
    hot.add_child(coffee)
    print(tree)


if __name__=='__main__':
    # test_linked_list()
    tree()