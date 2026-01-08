vector<int> numsSameConsecDiff(int n, int k) {
	vector<int> ret;
	vector<int> digits(n,0);
	
	for (int i = 1; i < 10; ++i)
	{
		digits[0] = i;
		backtrack(ret, digits, n, k, 0);
	}
	
	return ret;
}

void backtrack(vector<int>& ret, vector<int>& digits, int n, int k, int currIndex)
{
	if (currIndex == n-1)
	{
		ret.push_back(vectorToInt(digits));
		return;
	}
	
	if (digits[currIndex]-k >= 0)
	{
		digits[currIndex+1] = digits[currIndex]-k;
		backtrack(ret, digits, n, k, currIndex+1);
	}
	
	if (k != 0 && digits[currIndex]+k < 10)
	{
		digits[currIndex+1] = digits[currIndex]+k;
		backtrack(ret, digits, n, k, currIndex+1);
	}
}

int vectorToInt(vector<int>& digits)
{
	int ret = 0;
	for (int i = 0; i < digits.size(); ++i)
	{
		ret *= 10;
		ret += digits[i];
	}
	return ret;
}