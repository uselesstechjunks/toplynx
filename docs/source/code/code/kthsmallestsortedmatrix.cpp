struct Node
{
	Node() {}
	Node(int _value, int _row, int _col) : value(_value), row(_row), col(_col) {}
	int value;
	int row;
	int col;
};

int kthSmallest(vector<vector<int>>& matrix, int k) {
	auto cmp = [](Node a, Node b)
	{
		return a.value > b.value;
	};
	priority_queue<Node,vector<Node>,decltype(cmp)> heap(cmp);
	for (int i = 0; i < matrix.size(); ++i)
		heap.push(Node(matrix[0][i], 0, i));
	for (int i = 1; i < k; ++i)
	{
		Node node = heap.top();
		heap.pop();
		if (node.row < matrix.size()-1)
		{
			node.value = matrix[node.row+1][node.col];
			node.row = node.row+1;
			heap.push(node);
		}
	}
	return heap.top().value;
}