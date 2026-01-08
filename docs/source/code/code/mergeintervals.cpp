vector<vector<int>> merge(vector<vector<int>>& intervals) {
	auto cmp = [](const vector<int>& i1, const vector<int>& i2)
	{
		if (i1[0] == i2[0])
		{
			return i1[1] < i2[1];
		}
		return i1[0] < i2[0];
	};

	sort(intervals.begin(), intervals.end(), cmp);
	vector<vector<int>> ret;
	ret.push_back(intervals[0]);

	for (size_t i = 1; i < intervals.size(); ++i)
	{
		vector<int>& last = ret.back();
		const vector<int>& curr = intervals[i];

		if (last[1] >= curr[0])
		{
			last[0] = min(last[0], curr[0]);
			last[1] = max(last[1], curr[1]);
		}
		else
		{
			ret.push_back(curr);
		}
	}

	return ret;
}
