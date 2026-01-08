// left size returns
// (a) p (if p is on the left) or 
// (b) lca (if it is on the left) or 
// (c) nullptr (if p is not found)
// right size returns 
// (a) q (if q is on the right) or 
// (b) lca (if it is on the right) or 
// (c) nullptr (if p is not found)
//
// lca cannot come from both sides - if lca comes from left, then right must return nullptr
// therefore, if non-null return comes from both sides, then it must be case (a) - then root becomes lca
// else just propagame whatever return value is coming from left/right 
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	if (root == nullptr || root == p || root == q)
		return root;
	
	TreeNode *left = lowestCommonAncestor(root->left, p, q);
	TreeNode *right = lowestCommonAncestor(root->right, p, q);
	
	if (left != nullptr && right != nullptr)
		return root;
	if (left != nullptr)
		return left;
	return right;
}