bool validPalindrome(string s) {
	if (s.size() < 2) return true;
	int l = 0;
	int r = s.size() - 1;
	while (l < r)
	{
		if (s[l] != s[r])
		{
			if (isPalindrome(s, l+1, r) || isPalindrome(s, l, r-1))
				return true;
			else
				return false;
		}
		++l;
		--r;
	}
	return true;
}
bool isPalindrome(const string& s, int l, int r)
{
	while (l < r)
	{
		if (s[l] != s[r]) 
			return false;
		++l;
		--r;
	}
	return true;
}