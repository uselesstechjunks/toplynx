def verifyPreorder(self, preorder: List[int]) -> bool:
	# the idea is to use a monotonic stack
	# to simulate call stack in a bst
	"""
	IMPORTANT
	BST assumption for preorder is that once a node is processed
	all the left subtree nodes are also processed. So the only values
	we're allowed to process are larger than the node.
	"""
	stack = []
	# all next values should be larger than this
	min_limit = -inf
	for curr in preorder:
		while stack and stack[-1] < curr:
			# updating the min_limit here because nothing that
			# comes after should be smaller than this
			min_limit = stack.pop()
		if min_limit > curr:
			return False
		stack.append(curr)
	return True