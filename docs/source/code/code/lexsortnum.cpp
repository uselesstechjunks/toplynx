vector<int> lexicalOrder(int n) 
{
	vector<int> ret;
	backtrack(1, n, ret);
	return ret;
}

// for 13
// 1->10->(100)->11->(110)->12->(120)->13->(130)->(14)->2->(20)->3->(30)->4->(40)
void backtrack(int curr, int n, vector<int>& ret)
{
	if (curr > n)
		return;

	ret.push_back(curr);
	backtrack(curr * 10, n, ret);

	if (curr % 10 != 9)
		backtrack(curr + 1, n, ret);
}