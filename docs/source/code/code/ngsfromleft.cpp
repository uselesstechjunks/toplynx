vector<int> nextGreaterElement(vector<int>& values)
{
	vector<int> ans(values.size());
	stack<int> st;
	
	for (int i = 0; i < values.size(); ++i)
	{
		while (!st.empty() && values[st.top()] < values[i])
		{
			ans[st.top()] = i;
			st.pop();
		}
		st.push(i);
	}
	
	while (!st.empty())
	{
		ans[st.top()] = -1;
		st.pop();
	}
	
	return ans;
}