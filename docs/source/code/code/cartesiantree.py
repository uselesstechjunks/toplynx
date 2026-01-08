def cartesianTree(self, nums: List[int]) -> Optional[TreeNode]:
	stack = []

	for num in nums:
		last_popped = None
		while stack and stack[-1].val < num:
			last_popped = stack.pop()

		curr_node = TreeNode(num)
		curr_node.left = last_popped

		if stack:
			stack[-1].right = curr_node

		stack.append(curr_node)

	return stack[0] if stack else None