class WordLadder:
	def preprocess(self, wordList: Set[str]) -> None:
			# word -> pattern
			# pattern -> words
			self.word_to_pattern : Dict[str, List[str]] = defaultdict(list)
			self.pattern_to_word : Dict[str, Set[str]] = defaultdict(set)

			for word in wordList:
				for i in range(len(word)):
					pattern = word[:i] + '*' + word[i+1:]
					self.word_to_pattern[word].append(pattern)
					self.pattern_to_word[pattern].add(word)
		
	def process(self, queue_curr, visited_curr, visited_other):
		curr_word = queue_curr.popleft()
		for pattern in self.word_to_pattern[curr_word]:
			for next_word in self.pattern_to_word[pattern]:
				if next_word in visited_other:
					self.distance = visited_curr[curr_word] + visited_other[next_word]
					return
				if next_word not in visited_curr:
					visited_curr[next_word] = visited_curr[curr_word] + 1
					queue_curr.append(next_word)

	def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
		if beginWord == endWord:
			return 0
		if not beginWord or not endWord or not wordList:
			return 0

		self.beginWord = beginWord
		self.endWord = endWord
		
		# create adjacency list
		wordList = set(wordList)
		if endWord not in wordList:
			return 0

		wordList.add(beginWord)
		self.preprocess(wordList)

		# bidirectional bfs from source and target
		queue_forward = deque([beginWord])
		queue_backward = deque([endWord])

		# visited keeps track of the depth in bfs trees
		visited_forward = {beginWord : 1}
		visited_backward = {endWord : 1}

		self.distance = inf

		while queue_forward and queue_backward:
			if len(queue_forward) < len(queue_backward):
				self.process(queue_forward, visited_forward, visited_backward)
			else:
				self.process(queue_backward, visited_backward, visited_forward)
			if self.distance < inf:
				return self.distance

		return 0