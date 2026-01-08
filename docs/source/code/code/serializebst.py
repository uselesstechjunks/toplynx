# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

	def serialize(self, root: Optional[TreeNode]) -> str:
		"""Encodes a tree to a single string.
		"""
		# note that just the preorder traversal is sufficient for serialization
		preorder = self.iterative_preorder_traversal(root)
		return ','.join([str(x) for x in preorder])

	def deserialize(self, data: str) -> Optional[TreeNode]:
		"""Decodes your encoded data to tree.
		"""
		if len(data) == 0:
			return None

		preorder = [int(x) for x in data.split(',')]
		return self.construct_from_preorder(preorder)
	
	def construct_from_preorder(self, preorder: List[int]) -> Optional[TreeNode]:
		assert(len(preorder) > 0)

		# root needs special attention as it has to be returned
		root = TreeNode(preorder[0])
		max_stack = [root]

		for index in range(1, len(preorder)):
			node = TreeNode(preorder[index])

			last_popped = None
			while max_stack and max_stack[-1].val < node.val:
				last_popped = max_stack.pop()

			if last_popped is not None:
				last_popped.right = node
			elif max_stack:
				max_stack[-1].left = node

			max_stack.append(node)
		
		return root

	def iterative_preorder_traversal(self, root: Optional[TreeNode]) -> List[int]:
		preorder = []

		stack = [root] if root else []
		while stack:
			node = stack.pop()
			preorder.append(node.val)
			if node.right:
				stack.append(node.right)
			if node.left:
				stack.append(node.left)
		
		return preorder

# Your Codec object will be instantiated and called as such:
# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# tree = ser.serialize(root)
# ans = deser.deserialize(tree)
# return ans