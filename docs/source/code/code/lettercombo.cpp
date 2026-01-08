class Solution {
public:
	Solution()
	{
		map.insert({'2',vector<char>({'a','b','c'})});
		map.insert({'3',vector<char>({'d','e','f'})});
		map.insert({'4',vector<char>({'g','h','i'})});
		map.insert({'5',vector<char>({'j','k','l'})});
		map.insert({'6',vector<char>({'m','n','o'})});
		map.insert({'7',vector<char>({'p','q','r','s'})});
		map.insert({'8',vector<char>({'t','u','v'})});
		map.insert({'9',vector<char>({'w','x','y','z'})});
	}

	vector<string> letterCombinations(string digits) {
		vector<string> solution;
		string s;
		if (digits.size() > 0)
			backtrack(digits, s, solution, 0);
		return solution;
	}

	void backtrack(string& digits, string s, vector<string>& solution, int k)
	{
		if (k == digits.size())
			solution.push_back(s);
		else
		{
			const vector<char>& candidates = map[digits[k]];
			for (char candidate : candidates)
			{
				backtrack(digits, s + candidate, solution, k + 1);
			}
		}
	}

	unordered_map<char,vector<char>> map;
};