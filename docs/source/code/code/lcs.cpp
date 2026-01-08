int longestCommonSubsequence(string text1, string text2) {
	vector<vector<int>> t(text1.size() + 1);
	for (int i = 0; i < t.size(); ++i)
	{
		t[i].resize(text2.size() + 1);
		fill(t[i].begin(), t[i].end(), 0);
	}
	
	for (int i = 0; i < text1.size(); ++i)
	{
		for (int j = 0; j < text2.size(); ++j)
		{
			if (text1[i] == text2[j])
			{
				t[i+1][j+1] = 1+t[i][j];
			}
			else
			{
				t[i+1][j+1] = max(t[i][j+1], t[i+1][j]);
			}
		}
	}
	
	return t[text1.size()][text2.size()];
}