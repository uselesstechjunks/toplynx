vector<int> nextGreaterElement(vector<int>& values)
{
	vector<int> ans(values.size());
	stack<int> st;
	
	for (int i = values.size()-1; i >= 0; --i)
	{
		while (!st.empty() && values[st.top()] < values[i])
		{
			st.pop();
		}
		if (!st.empty())
			ans[i] = -1;
		else
			ans[i] = st.top();
		st.push(i);
	}
	
	return ans;
}