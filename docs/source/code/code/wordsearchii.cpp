class Solution {
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        TrieNode* root = populate(words);
        vector<string> res;
        string buf(10, 0);
        
        for (int i = 0; i < board.size(); ++i)
        {
            for (int j = 0; j < board[i].size(); ++j)
            {
                backtrack(res, board, root, buf, 0, i, j, words.size());
            }
        }
        
        return res;
    }
private:    
    void backtrack(vector<string>& res, vector<vector<char>>& board, TrieNode* node, string buf, int k, int i, int j, size_t max)
    {
        if (i < 0 || i >= board.size() || j < 0 || j >= board[i].size() || !node->contains(board[i][j]) || board[i][j] == 0 || res.size() == max)
        {
            return;
        }
        
        char orig = board[i][j];
        board[i][j] = 0;
        buf[k] = orig;

        // !!IMPORTANT!! add the word before calling
        TrieNode* child = node->get(orig);
        if (child->isEnd())
        {
            child->setEnd(false);
            string curr = buf.substr(0,k+1);
            res.push_back(curr);
        }

        backtrack(res, board, child, buf, k+1, i+1, j, max);
        backtrack(res, board, child, buf, k+1, i-1, j, max);
        backtrack(res, board, child, buf, k+1, i, j+1, max);
        backtrack(res, board, child, buf, k+1, i, j-1, max);

        board[i][j] = orig;
    }

    TrieNode* populate(const vector<string>& words)
    {
        TrieNode* root = new TrieNode();
        for (const auto& word : words)
        {
            TrieNode* node = root;
            for (const auto& c : word)
            {
                if (!node->contains(c))
                {
                    node->set(c);
                }
                node = node->get(c);
            }
            node->setEnd(true);
        }
        return root;
    }
};

class TrieNode
{
public:
    TrieNode() : children(vector<TrieNode*>(26,nullptr)), end(false) {}
    ~TrieNode()
    {
        for (int i = 0; i < children.size(); ++i)
        {
            delete children[i];
        }
    }
    void set(char c)
    {
        if (c >= 'a' && c <= 'z' && children[c-'a'] == nullptr)
        {
            children[c-'a'] = new TrieNode();
        }
    }
    bool contains(char c) const
    {
        return get(c) != nullptr;
    }
    TrieNode* get(char c) const
    {
        if (c >= 'a' && c <= 'z')
        {
            return children[c-'a'];
        }
        return nullptr;
    }
    void setEnd(bool value)
    {
        end = value;
    }
    bool isEnd() const
    {
        return end;
    }
private:
    vector<TrieNode*> children;
    bool end;
};