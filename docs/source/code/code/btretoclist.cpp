Node *bTreeToCList(Node *root)
{
	if (root == nullptr) return root;
	pair<Node*, Node*> ret = helper(root);
	ret.second->right = ret.first;
	ret.first->left = ret.second;
	return ret.first;
}

pair<Node*,Node*> helper(Node* root)
{
	if (root == nullptr) return {root,root};
	if (root->left == nullptr && root->right == nullptr)
	{
		return {root,root};
	}
	auto left = helper(root->left);
	auto right = helper(root->right);
	Node* first = left.first;
	Node* last = right.second;
	if (left.first == nullptr)
	{
		first = root;
	}
	if (left.second != nullptr)
	{
		left.second->right = root;
		root->left = left.second;
	}
	if (right.second == nullptr)
	{
		last = root;
	}
	if (right.first != nullptr)
	{
		right.first->left = root;
		root->right = right.first;
	}
	return {first, last};
}