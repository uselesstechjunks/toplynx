bool exist(vector<vector<char>>& board, string word) {
	for (int i = 0; i < board.size(); ++i)
	{
		for (int j = 0; j < board[i].size(); ++j)
		{
			// the key here is to not store the current forming word but use index to refer 
			// to the original word. this way we can prune early when characters mismatch			
			int idx = 0;
			if (backtrack(board, word, i, j, idx))
			{
				return true;
			}
		}
	}
	
	return false;
}

bool backtrack(vector<vector<char>>& board, const string& word, int i, int j, int idx)
{
	if (idx == word.size())
	{
		return true;
	}
	if (i < 0 || i >= board.size() || j < 0 || j >= board[i].size() || board[i][j] == 0 || board[i][j] != word[idx])
	{
		return false;
	}
	
	char c = board[i][j];
	board[i][j] = 0; // mark as visited
	
	bool ret =	backtrack(board, word, i-1, j, idx+1) || 
				backtrack(board, word, i+1, j, idx+1) || 
				backtrack(board, word, i, j-1, idx+1) || 
				backtrack(board, word, i, j+1, idx+1);
	
	board[i][j] = c; // restore the visited node
	return ret;
}