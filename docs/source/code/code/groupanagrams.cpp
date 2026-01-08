vector<vector<string>> groupAnagrams(vector<string>& strs) {
	vector<vector<string>> res;
	if (strs.size() == 0)
		return res;
	
	unordered_map<string,vector<string>> ht;
	for (int i = 0; i < strs.size(); ++i)
	{
		string key = strs[i];
		sort(key.begin(), key.end());
		if (ht.find(key) != ht.end())
		{
			ht[key].push_back(strs[i]);
		}
		else
		{
			vector<string> v(1);
			v[0] = strs[i];
			ht.insert({key, v});
		}
	}
	for (const auto& iter : ht)
	{
		res.push_back(iter.second);
	}
	
	return res;
}