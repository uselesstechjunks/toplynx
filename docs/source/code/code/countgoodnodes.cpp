int goodNodes(TreeNode* root) {
	if (root == nullptr)
		return 0;
	int count = 0;
	helper(root, root->val, count);
	return count;
}

void helper(TreeNode* root, int maxSoFar, int& count)
{
	if (root == nullptr)
		return;
	
	if (root->val >= maxSoFar)
	{
		++count;
		maxSoFar = root->val;
	}
	
	helper(root->left, maxSoFar, count);
	helper(root->right, maxSoFar, count);
}