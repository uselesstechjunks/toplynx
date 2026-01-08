vector<string> wordBreak(string s, vector<string>& wordDict) {
	vector<string> solution;
	unordered_set<string> dict(wordDict.size());
	for (const auto& word : wordDict)
	{
		dict.insert(word);
	}
	string partial;
	backtrack(s, partial, dict, solution, 0);
	return solution;
}

void backtrack(string& s, string partial, unordered_set<string>& dict, vector<string>& solution, int k)
{
	if (k == s.size())
	{
		partial = partial.substr(0, partial.size()-1);
		solution.push_back(partial);
	}
	else
	{
		string word;
		for (int i = k; i < s.size(); ++i)
		{
			word += s[i];
			if (dict.find(word) != dict.end()) 
			{
				backtrack(s, partial + word + " ", dict, solution, i+1);
			}
		}
	}
}