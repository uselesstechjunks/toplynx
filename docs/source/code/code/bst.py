from typing import Optional
from random import shuffle

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    def __repr__(self):
        return f'TreeNode(val={self.val}),[left={self.left}],[right={self.right}]'

class BST:
    def __init__(self):
        self.root = None

    def find(self, val:int) -> Optional[TreeNode]:
        def find_impl(root, val):
            if root is None: return None
            if root.val == val: return root
            return find_impl(root.left, val) if val < root.val else find_impl(root.right, val)
        return find_impl(self.root, val)

    def min(self) -> Optional[TreeNode]:
        def min_impl(root):
            if root is None: return None
            if root.left is None and root.right is None: return root
            return min_impl(root.left)
        return min_impl(self.root)

    def max(self) -> Optional[TreeNode]:
        def max_impl(root):
            if root is None: return None
            if root.left is None and root.right is None: return root
            return max_impl(root.right)
        return max_impl(self.root)
    
    def remove(self, val:int):
        print(f'removing {val}')
        def remove_impl(root, val):
            if root is None:
                return None
            if val < root.val:
                root.left = remove_impl(root.left, val)
            elif root.val < val:
                root.right = remove_impl(root.right, val)
            else:
                if root.right is None:
                    return root.left
                prev, node = root.right, root.right.left
                if node is None:
                    prev.left = root.left
                    root = prev
                else:
                    while node.left is not None:
                        prev = node
                        node = node.left
                    prev.left = node.right
                    node.right = root.right
                    node.left = root.left
                    root = node
            return root
        self.root = remove_impl(self.root, val)
        return self.root

    def add(self, val:int):
        def add_impl(root, val):
            if root is None:
                return TreeNode(val)
            if val < root.val:
                root.left = add_impl(root.left, val)
            elif root.val < val:
                root.right = add_impl(root.right, val)
            return root
        return add_impl(self.root, val)

def test1():
    root = TreeNode(5)
    root.left = TreeNode(2)
    root.left.left = TreeNode(1)
    root.right = TreeNode(7)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(8)

    bst = BST()
    bst.root = root
    print(f'min={bst.min()}, max={bst.max()}')

    for i in range(10):
        if bst.find(i) is None:
            bst.add(i)
        print(f'min={bst.min()}, max={bst.max()},bst={bst.find(i)}')

    values = list(range(10))
    shuffle(values)
    for i in values:
        bst.remove(i)
        print(f'bst={bst.root}')

if __name__ == '__main__':
    test1()