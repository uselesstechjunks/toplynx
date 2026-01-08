vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
	vector<vector<int>> solution;
	if (root == nullptr)
		return solution;
	
	vector<int> path;
	path.resize(5000);
	path[0] = root->val;
	
	backtrack(root, targetSum, root->val, solution, path, 0);
	
	return solution;
}

void backtrack(TreeNode* root,int targetSum,int runningSum,vector<vector<int>>& solution,vector<int>& path,int lastIndex)
{
	if (is_leaf(root) && targetSum == runningSum)
	{
		vector<int> copy = path;
		copy.resize(lastIndex+1);
		solution.push_back(copy);
	}
	
	if (root->left != nullptr)
	{
		path[lastIndex+1] = root->left->val;
		backtrack(root->left, targetSum, runningSum+root->left->val, solution, path, lastIndex+1);
	}
	
	if (root->right != nullptr)
	{
		path[lastIndex+1] = root->right->val;
		backtrack(root->right, targetSum, runningSum+root->right->val, solution, path, lastIndex+1);
	}
}

bool is_leaf(TreeNode* root)
{
	return root->left == nullptr && root->right == nullptr;
}