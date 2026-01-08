class LRUCache:

	def __init__(self, capacity: int):
		self.capacity = capacity
		self.ordered_dict = collections.OrderedDict()

	def get(self, key: int) -> int:
		if key not in self.ordered_dict:
			return -1
		
		# every time we use a value, we move it to the last
		self.ordered_dict.move_to_end(key)
		return self.ordered_dict[key]

	def put(self, key: int, value: int) -> None:
		if key in self.ordered_dict:
			self.ordered_dict.move_to_end(key)
		
		self.ordered_dict[key] = value
		if len(self.ordered_dict) > self.capacity:
			# False means the first element is popped
			self.ordered_dict.popitem(False)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)