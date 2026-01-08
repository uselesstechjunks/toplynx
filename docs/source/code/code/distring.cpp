string smallestNumber(string pattern) 
{
	string ret(pattern.size()+1, 0);
	vector<bool> used(10, false);

	// instead of simulating all possible cases, here we systematically explore only those paths
	// that would lead to the solution when the flow reaches the leaf for the first time.
	// so need a flag to indicate early return as soon as a leaf node in the recursion tree is found
	bool found = false;
	// since pattern starts from 2nd digit, it's a better idea to pull out the iteration
	// for the first digit into a separate part instead of making it part of the backtrack method
	for (int n = 1; n < 10 && !found; ++n)
	{
		ret[0] = n;
		used[n] = true;
		found = backtrack(0, pattern, ret, used);
		used[n] = false;
	}

	for (int n = 0; n < ret.size(); ++n)
		ret[n] += '0';

	return ret;
}
    
bool backtrack(int k, const string& pattern, string& ret, vector<bool>& used)
{
	if (k == pattern.size())
		return true;

	bool found = false;
	if (pattern[k] == 'I')
	{
		for (int n = ret[k]+1; n < 10 && !found; ++n)
		{
			if (!used[n])
			{
				ret[k+1] = n;
				used[n] = true;
				found = backtrack(k+1, pattern, ret, used);
				used[n] = false;
			}
		}
	}
	else
	{
		for (int n = ret[k]-1; n > 0 && !found; --n)
		{
			if (!used[n])
			{
				ret[k+1] = n;
				used[n] = true;
				found = backtrack(k+1, pattern, ret, used);
				used[n] = false;
			}
		}
	}
	return found;
}