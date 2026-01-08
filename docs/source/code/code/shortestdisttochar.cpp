vector<int> shortestToChar(string s, char c) {
	vector<int> ret(s.size(), s.size());
	
	// update the distance from left occurance
	int pos = -s.size();
	for (int i = 0; i < s.size(); ++i)
	{
		if (s[i] == c)
			pos = i;
		
		ret[i] = i - pos;
	}
	
	// update the distance from right occurance
	for (int i = pos-1; i >= 0; --i)
	{
		if (s[i] == c)
			pos = i;
		
		ret[i] = min(ret[i], pos - i);
	}
	
	return ret;
}