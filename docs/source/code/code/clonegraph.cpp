/*
// Definition for a Node.
class Node {
public:
	int val;
	vector<Node*> neighbors;
	Node() {
		val = 0;
		neighbors = vector<Node*>();
	}
	Node(int _val) {
		val = _val;
		neighbors = vector<Node*>();
	}
	Node(int _val, vector<Node*> _neighbors) {
		val = _val;
		neighbors = _neighbors;
	}
};
*/

Node* cloneGraph(Node* node) {
	unordered_map<Node*,Node*> map;
	Node* clone = dfs(node, map);
	return clone;
}

Node* dfs(Node* node, unordered_map<Node*,Node*>& map)
{
	Node* cloneNode = nullptr;
	if (node == nullptr) 
		return cloneNode;
	
	if (map.find(node) == map.end())
		map.insert({node, new Node(node->val)});
	
	cloneNode = map[node];
	cloneNode->neighbors.resize(node->neighbors.size());
	
	for (int i = 0; i < node->neighbors.size(); ++i)
	{
		if (map.find(node->neighbors[i]) == map.end())
			cloneNode->neighbors[i] = dfs(node->neighbors[i], map);
		else
			cloneNode->neighbors[i] = map[node->neighbors[i]];
	}
	
	return cloneNode;
}