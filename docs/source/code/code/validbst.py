def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def impl(root, l, r):
            if root is None: 
                return True
            if root.val < l or root.val > r:
                return False
            return impl(root.left, l, root.val-1) and impl(root.right, root.val+1, r)
        return impl(root, float('-inf'), float('inf'))