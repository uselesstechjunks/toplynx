void backtrack(vector<int>& nums, int k, vector<vector<int>>& solution)
{
	if (k == nums.size())
		solution.push_back(nums);
	else
	{
		for (int i = k; i < nums.size(); ++i)
		{
			swap(nums[k], nums[i]);
			backtrack(nums, k+1, solution);
			swap(nums[k], nums[i]);
		}
	}
}