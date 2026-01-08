from random import shuffle
from typing import List

class BinaryIndexedTree2D:

	def __init__(self, matrix: List[List[int]]):
		def build():
			for i in range(self.m):
				for j in range(self.n):
					self.add(i, j, self.matrix[i][j])

		self.m, self.n = len(matrix), len(matrix[0]) if len(matrix) > 0 else 0
		self.matrix = matrix
		self.bit = [[0] * self.n for _ in range(self.m)]
		build()
		print(self.bit)
	
	def nextIndex(self, index):
		return index | (index + 1)
	
	def prevIndex(self, index):
		return (index & (index + 1)) - 1
	
	def add(self, rowIndex, colIndex, delta):
		i = rowIndex
		while i < self.m:
			j = colIndex
			while j < self.n:
				self.bit[i][j] += delta
				j = self.nextIndex(j)
			i = self.nextIndex(i)

	def sum(self, lastRowIndex, lastColIndex):
		res = 0
		i = lastRowIndex
		while i >= 0:
			j = lastColIndex
			while j >= 0:
				res += self.bit[i][j]
				j = self.prevIndex(j)
			i = self.prevIndex(i)
		return res

	def update(self, row: int, col: int, val: int) -> None:
		delta = val - self.matrix[row][col]
		self.matrix[row][col] = val
		self.add(row, col, delta)

	def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
		topLeft = self.sum(row1-1, col1-1)
		topRight = self.sum(row1-1, col2) - topLeft
		botLeft = self.sum(row2, col1-1) - topLeft
		botRight = self.sum(row2, col2) - topLeft - topRight - botLeft
		return botRight