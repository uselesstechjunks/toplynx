bool isValid(string s) {
	if (s.size() == 0) 
		return true;

	stack<char> st;
	for (int i = 0; i < s.size(); ++i)
	{
		if (!st.empty())
		{
			if (matches(st.top(), s[i]))
				st.pop();
			else
				st.push(s[i]);
		}
		else
		{
			st.push(s[i]);
		}
	}
	return st.empty();
}

bool matches(char a, char b)
{
	if ((a == '(' && b == ')') || (a == '{' && b == '}') || (a == '[' && b == ']'))
		return true;
	return false;
}