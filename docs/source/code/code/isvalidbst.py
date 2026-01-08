def isValidBST(self, root: Optional[TreeNode]) -> bool:
	def range(root, tree_min, tree_max) -> bool:
		if not root:
			return True
		
		if tree_min < root.val and root.val < tree_max:
			# root is valid, now check subtrees
			left_valid = range(root.left, tree_min, root.val)
			right_valid = range(root.right, root.val, tree_max)
			if left_valid and right_valid:
				return True
		
		return False

	def minmax(root) -> Tuple[int,int]:
		# returns min, max of the tree rooted at root
		if not root:
			# note that the order is kept like this
			# so that any parent subtree is not invalidated because
			# of empty children
			return inf, -inf
		
		left_min, left_max = minmax(root.left)
		right_min, right_max = minmax(root.right)

		if left_max < root.val and root.val < right_min:
			left_min = min(left_min, root.val)
			right_max = max(right_max, root.val)
			return left_min, right_max
		
		# if comes here, then all parent subtrees are invalidated
		return -inf, inf
	
	# tree_min, tree_max = minmax(root)
	# return tree_min > -inf and tree_max < inf

	return range(root, -inf, inf)