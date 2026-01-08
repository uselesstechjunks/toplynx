int minDepth(TreeNode<int>* root)
{
	if (root == nullptr) return 0;
	if (root->left == nullptr && root->right == nullptr) return 1;
	int left = minDepth(root->left);
	int right = minDepth(root->right);
	if (left == 0 && right != 0)
		return right + 1;
	if (left != 0 && right == 0)
		return left + 1;
	return min(left, right) + 1;
}