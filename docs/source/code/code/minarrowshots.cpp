int findMinArrowShots(vector<vector<int>>& points) 
{
	auto cmp = [](const vector<int>& a, const vector<int>& b)
	{
		return (a[1] < b[1]);
	};

	sort(points.data(), points.data()+points.size(), cmp);
	pair<int,int> top = {points[0][0], points[0][1]};
	int count = 1;

	for (size_t i = 1; i < points.size(); ++i)
	{
		pair<int,int> curr({points[i][0], points[i][1]});
		
		if (top.second >= curr.first)
		{
			if (top.second > curr.second)
			{
				top = {curr.first, curr.second};
			}
			else
			{
				top = {curr.first, top.second};
			}
		}
		else
		{
			++count;
			top = {curr.first, curr.second};
		}
	}

	return count;
}
