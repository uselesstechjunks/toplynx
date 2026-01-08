vector<string> letterCasePermutation(string s) 
{
	vector<string> ret;
	backtrack(ret, s, 0);
	return ret;
}

void backtrack(vector<string>& ret, string& s, int k)
{
	if (k == s.size())
	{
		ret.push_back(s);
		return;
	}

	// recurse once - applies to both letters and digit
	backtrack(ret, s, k+1);
	
	// if it's a letter, then change case and recurse once more
	if (isLowercase(s[k]) || isUppercase(s[k]))
	{
		s[k] = changeCase(s[k]);
		backtrack(ret, s, k+1);
		s[k] = changeCase(s[k]);
	}
}

bool isLowercase(char c)
{
	if (c >= 'a' && c <= 'z')
		return true;
	return false;
}

bool isUppercase(char c)
{
	if (c >= 'A' && c <= 'Z')
		return true;
	return false;
}

char changeCase(char c)
{
	if (isLowercase(c))
		return c - ('a'-'A');
	else if (isUppercase(c))
		return c + ('a'-'A');
	return c;
}