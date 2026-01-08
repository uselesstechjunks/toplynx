def correctBinaryTree(self, root: TreeNode) -> TreeNode:
	# idea is to use bfs to traverse nodes at the same level
	"""
	# since the problematic node points to right, we traverse
	# from right to left in each level.
	"""
	# whenever a node's right is pointing towards something which
	# is already visited, we know that the node is problematic.
	"""
	since we'd also have to delete this node, we keep track of
	the parent pointers in the queue itself so that we can delete
	it as soon as we find it and don't need to traverse the tree again
	"""
	queue = deque([(root, None)])

	while queue:
		size = len(queue)
		# keeps track of the visited nodes only at current level
		visited = set()

		for _ in range(size):
			node, parent = queue.popleft()

			if node.right and node.right.val in visited:
				if node == parent.left:
					parent.left = None
				else:
					parent.right = None
				return root
			
			visited.add(node.val)

			if node.right:
				queue.append((node.right, node))
			if node.left:
				queue.append((node.left, node))

	return root