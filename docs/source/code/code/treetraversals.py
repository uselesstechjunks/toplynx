def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
	def dfs(root):
		nonlocal result
		if not root:
			return
		result.append(root.val)
		dfs(root.left)
		dfs(root.right)
	
	def iter(root):
		nonlocal result
		stack = [root] if root else []
		while stack:
			# stack top == current call stack
			node = stack.pop()
			result.append(node.val)

			""" remmeber to push right first """
			if node.right:
				stack.append(node.right)
			if node.left:
				stack.append(node.left)

def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
	def dfs(root):
		nonlocal result
		if not root:
			return
		dfs(root.left)
		result.append(root.val)
		dfs(root.right)
	
	def push_left(stack, root):
		while root:
			stack.append(root)
			root = root.left
	
	def iter(root):
		nonlocal result
		"""
		we need to simulate how the call stack works
		(a) push as many left elements as possible as init
		(b) popping means processing
		(c) once we pop, we need to move right and then do (a) again
		"""
		stack = []
		""" NOTE we have to do this before entering the loop """
		push_left(stack, root)

		while stack:
			node = stack.pop()
			result.append(node.val)
			push_left(stack, node.right)

	result = []
	iter(root)
	return result

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:

	def __init__(self, root: Optional[TreeNode]):
		self.stack = []
		self.push_left(root)

	def push_left(self, node):
		while node:
			self.stack.append(node)
			node = node.left

	def next(self) -> int:
		top = self.stack.pop()
		self.push_left(top.right)
		return top.val

	def hasNext(self) -> bool:
		return len(self.stack) > 0


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
