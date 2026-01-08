vector<string> generateParenthesis(int n) {
	vector<string> res;
	if (n == 0)
		return res;
	
	backtrack(res, n, 0, 0, "");
	return res;
}

void backtrack(vector<string>& res, int n, int leftCount, int rightCount, string str)
{
	if (leftCount == n && rightCount == n)
		res.push_back(str);
	else
	{
		// the key idea is to prune the backtrack tree for impossible moves
		// left move is possible when leftCount < n
		// right move is possible when leftCount > rightCount
		if (leftCount < n)
			backtrack(res, n, leftCount + 1, rightCount, str + "(");
		if (leftCount > rightCount)
			backtrack(res, n, leftCount, rightCount + 1, str + ")");
	}
}