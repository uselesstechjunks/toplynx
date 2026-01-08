vector<vector<int>> levelOrder(TreeNode* root) {
	vector<vector<int>> res;
	if (root == nullptr)
		return res;
	
	queue<TreeNode*> q;
	q.push(root);
	
	while (!q.empty())
	{
		int size = q.size();
		vector<int> currentLevel(size);
		for (int i = 0; i < size; ++i)
		{
			TreeNode* curr = q.front();
			q.pop();
			currentLevel[i] = curr->val;
			if (curr->left != nullptr)
				q.push(curr->left);
			if (curr->right != nullptr)
				q.push(curr->right);
		}
		res.push_back(currentLevel);
	}
	
	return res;
}