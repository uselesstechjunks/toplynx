int largestRectangleArea(vector<int>& heights) {
	vector<int> pse(heights.size());
	vector<int> nse(heights.size());
	stack<int> st;
	for (int i = 0; i < heights.size(); ++i)
	{
		while (!st.empty() && heights[st.top()] >= heights[i])
			st.pop();
		if (!st.empty())
			pse[i] = st.top();
		st.push(i);
	}
	
	for (int i = 0; i < heights.size(); ++i)
	{
		while (!st.empty() && heights[st.top()] >= heights[i])
		{
			nse[st.top()] = i;
			st.pop();
		}
		st.push(i);
	}
	while (!st.empty())
	{
		nse[st.top()] = heights.size();
		st.pop();
	}
	int max_area = 0;
	for (int i = 0; i < heights.size(); ++i)
	{
		max_area = max(max_area, (nse[i]-pse[i]-1)*heights[i]);
	}
	return max_area;
}