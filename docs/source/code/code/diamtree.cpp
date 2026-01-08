int diameterOfBinaryTree(TreeNode* root) {
	int diam = 0;
	if (root == nullptr)
		return diam;
	
	helper(root, diam);
	return diam;
}

// returns height of the subtree
int helper(TreeNode* root, int& diam)
{
	if (root == nullptr)
		return 0;
	
	int leftHeight = helper(root->left, diam);
	int rightHeight = helper(root->right, diam);
	int currHeight = max(leftHeight, rightHeight) + 1;
	
	int diamThroughRoot = leftHeight + rightHeight;
	diam = max(diam, diamThroughRoot);
	
	return currHeight;
}