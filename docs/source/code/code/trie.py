class TrieNode:
	def __init__(self):
		self.desc = {}
		self.is_end = False
	def __repr__(self):
		return f'{self.desc}, {self.is_end}'

class WordDictionary:

	def __init__(self):
		self.root = TrieNode()

	def addWord(self, word: str) -> None:
		node = self.root
		# abc
		# root: {a: {b: {c: {}, True}, False}, False}, False
		for char in word:
			if char not in node.desc:
				node.desc[char] = TrieNode()
			node = node.desc[char]
		node.is_end = True
	
	def dfs(self, node, word, k) -> bool:
		while k < len(word):
			char = word[k]
			if char not in node.desc:
				if char is '.':
					for key in node.desc.keys():
						if self.dfs(node.desc[key], word, k+1):
							return True
				return False
			node = node.desc[char]
			k += 1
		return node.is_end

	def search(self, word: str) -> bool:
		return self.dfs(self.root, word, 0)