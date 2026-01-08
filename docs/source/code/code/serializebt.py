def deserialize(self, s: str) -> Optional[TreeNode]:
	def parse(s):
		if s == '':
			return None

		stack = []
		node = TreeNode()
		sign = 1

		for char in s:
			if char == '-':
				sign = -1
			elif char.isdigit():
				node.val = node.val * 10 + sign * int(char)
			else: # it's ( or )
				if char == '(':
					stack.append(node)
					node = TreeNode()
					sign = 1
				elif char == ')':
					parent = stack.pop()
					if not parent.left:
						parent.left = node
					else:
						parent.right = node
					node = parent # super important

		return node
	
	return parse(s)

def serialize(self, root: Optional[TreeNode]) -> str:
	def dfs(root):
		nonlocal result
		if not root:
			return
		result.append(str(root.val))
		if root.left or root.right:
			result.append('(')
			dfs(root.left)
			result.append(')')
		if root.right:
			result.append('(')
			dfs(root.right)
			result.append(')')
	result = []
	dfs(root)
	return ''.join(result)