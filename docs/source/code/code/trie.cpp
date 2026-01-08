class TrieNode
{
public:
	TrieNode() : end(false),children(vector<TrieNode*>(26,nullptr) {}
	TrieNode* get(char c) const
	{
		return children[c-'a'];
	}
	void set(char c)
	{
		children[c-'a'] = new TrieNode();
	}
	void setEnd()
	{
		end = true;
	}
	bool isEnd() const
	{
		return end;
	}
private:
	vector<TrieNode*> children;
	bool end;
};

class Trie 
{
public:
	Trie() 
	{
		root = new TrieNode();
	}

	void insert(string word) 
	{
		TrieNode* node = root;
		for (char c : word)
		{
			if (node->get(c) == nullptr)
				node->set(c);
			node = node->get(c);
		}
		node->setEnd(true);
	}

	bool search(string word) 
	{
		TrieNode* node = getNode(word);
		if (node != nullptr && node->isEnd())
			return true;
		return false;
	}

	bool startsWith(string prefix) 
	{
		TrieNode* node = getNode(prefix);
		if (node != nullptr)
			return true;
		return false;
	}
private:
	TrieNode* root;

	TrieNode* getNode(const string& word) const
	{
		TrieNode* node = root;
		for (char c : word)
		{
			node = node->get(c);
			if (node == nullptr)
				break;
		}
		return node;
	}
};

/**
* Your Trie object will be instantiated and called as such:
* Trie* obj = new Trie();
* obj->insert(word);
* bool param_2 = obj->search(word);
* bool param_3 = obj->startsWith(prefix);
*/