// vector<bool> flag(nums.size(), false);
// vector<vector<int>> solution;
// call: backtrack(nums, flag, solution, 0);
void backtrack(vector<int>& nums,vector<bool>& flag,vector<vector<int>>& solution,int k)
{
	if (k == nums.size())
		processSolution(nums, flag, solution);
	else
	{
		// every element can either be in the solution set (true) or not (false)
		flag[k] = false;
		backtrack(nums, flag, solution, k+1);
		flag[k] = true;
		backtrack(nums, flag, solution, k+1);
		flag[k] = false;
	}
}

void processSolution(vector<int>& nums, vector<bool>& flag, vector<vector<int>>& solution)
{
	vector<int> s;
	for (int i = 0; i < flag.size(); ++i)
	{
		if (flag[i])
			s.push_back(nums[i]);
	}
	solution.push_back(s);
}