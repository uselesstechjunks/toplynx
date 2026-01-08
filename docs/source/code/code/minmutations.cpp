int minMutation(string startGene, string endGene, vector<string>& bank) 
{
	// maps string to index in the bank
	// would save memory for intermediate nodes if we can refer by index in bank
	unordered_map<string, int> dict = populateDict(bank);
	
	vector<char> choices({'A','C','G','T'});

	// we can store indices from bank just by index
	unordered_set<int> visited;

	queue<int> q;
	q.push(-1); // -1 for the start gene as that doesn't exist in bank
	int path = 0;

	while (!q.empty())
	{
		// need level order traversal here because we need the path length
		int size = q.size();
		for (int i = 0; i < size; ++i)
		{
			int u = q.front();
			q.pop();

			visited.insert(u);

			string& curr = startGene;
			if (u != -1)
			{
				curr = bank[u];
			}

			if (curr == endGene)
			{
				return path;
			}

			// ensure to change one character at a time at any position of the gene
			for (int j = 0; j < curr.size(); ++j)
			{
				for (char c : choices)
				{
					char orig = curr[j];
					curr[j] = c;
					if (dict.find(curr) != dict.end() && visited.find(dict[curr]) == visited.end())
					{
						q.push(dict[curr]);
					}
					curr[j] = orig;
				}
			}
		}
		++path;
	}

	return -1;
}

unordered_map<string, int> populateDict(const vector<string>& bank)
{
	unordered_map<string, int> dict;
	for (int i = 0; i < bank.size(); ++i)
	{
		dict.insert({bank[i], i});
	}
	return dict;
}