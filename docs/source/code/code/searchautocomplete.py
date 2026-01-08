class TrieNode:
	def __init__(self):
		self.children = {}
		self.sentences = defaultdict(int)

class AutocompleteSystem:

	def __init__(self, sentences: List[str], times: List[int]):        
		self.root = TrieNode()
		self.curr_node = self.root
		self.current_sentence = []
		self.build(sentences, times)

	def add_sentence(self, sentence, count):
		""" adds a sentence to the trie and increases the count as specified by the parameter """
		node = self.root
		for c in sentence:
			if c not in node.children:
				node.children[c] = TrieNode()
			node = node.children[c]
			node.sentences[sentence] += count

	def build(self, sentences, times):
		for i, sentence in enumerate(sentences):
			self.add_sentence(sentence, times[i])

	def input(self, c: str) -> List[str]:
		# print(f'input={c}')
		if c == '#':
			sentence = ''.join(self.current_sentence)
			self.add_sentence(sentence, 1)
			self.curr_node = self.root
			self.current_sentence = []
			return []
		else:
			self.current_sentence.append(c)
			if c not in self.curr_node.children:
				self.curr_node.children[c] = TrieNode()
			self.curr_node = self.curr_node.children[c]
			# print(f'sentences={self.curr_node.sentences}')
			sorted_sentences = sorted(self.curr_node.sentences.keys(), key=lambda k:(-self.curr_node.sentences[k], k))
			# print(sorted_sentences)
			return sorted_sentences[:3]

	# Your AutocompleteSystem object will be instantiated and called as such:
	# obj = AutocompleteSystem(sentences, times)
	# param_1 = obj.input(c)