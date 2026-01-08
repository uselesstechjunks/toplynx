int maxPathSum(TreeNode* root) {
	int max = INT_MIN;
	maxPathSum(root, max);
	return max;
}

int maxPathSum(TreeNode* root, int& max) {
	if (root == nullptr) return 0;
	
	int left = maxPathSum(root->left, max);
	int right = maxPathSum(root->right, max);
	
	int maxValueThroughRoot = root->val;
	int maxValueSubTree = root->val;
	if (left <= 0 && right > 0) {
		maxValueThroughRoot += right;
		maxValueSubTree += right;
	} else if (left > 0 && right <= 0) {
		maxValueThroughRoot += left;
		maxValueSubTree += left;
	} else if (left > 0 && right > 0) {
		maxValueThroughRoot += std::max(left, right);
		maxValueSubTree += left + right;
	}
	max = std::max(max, maxValueSubTree);
	return maxValueThroughRoot;
}