def flatten(self, root: Optional[TreeNode]) -> None:
	"""
	Do not return anything, modify root in-place instead.
	"""
	def impl(root: Optional[TreeNode]) -> Tuple[Optional[TreeNode], Optional[TreeNode]]:
		"""
		returns first and last nodes of the flattened list [first...last]
		"""
		if not root:
			return root, root
		
		if not root.left and not root.right:
			return root, root
		
		left_first, left_last = impl(root.left)
		right_first, right_last = impl(root.right)

		root.left = None

		# NOTE left and right subtrees cannot both be empty 
		# since we've added a leaf check already

		# case 1: left subtre was empty
		if not left_first:
			left_last = root

		# case 2: right subtree was empty
		if not right_first:
			right_last = left_last
		
		# case 3: none of the subtrees was empty
		# NOTE because of the leaf check, this MUST mean that
		# both subtrees were non empty
		root.right = left_first
		left_last.right = right_first

		return root, right_last
	
	head, _ = impl(root)
	return head