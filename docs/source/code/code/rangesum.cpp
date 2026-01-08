int rangeSumBST(TreeNode* root, int low, int high) {
	if (root == nullptr)
		return 0;
	int sum = 0;
	helper(root, low, high, sum);
	return sum;
}

void helper(TreeNode* root, int low, int high, int& sum)
{
	if (root == nullptr) return;
	if (root->val > high)
		helper(root->left, low, high, sum);
	else if (root->val < low)
		helper(root->right, low, high, sum);
	else
	{
		helper(root->left, low, high, sum);
		helper(root->right, low, high, sum);
		sum += root->val;
	}
}