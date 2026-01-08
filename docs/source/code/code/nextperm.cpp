void nextPermutation(vector<int>& nums) {
	for (int i = nums.size()-1; i > 0; --i)
	{
		if (nums[i-1] < nums[i])
		{
			int lg_idx = i;
			for (int j = i+1; j < nums.size(); ++j)
			{
				if (nums[j] > nums[i-1] && nums[j] < nums[lg_idx])
					lg_idx = j;
			}
			swap(nums[i-1], nums[lg_idx]);
			sort(nums.begin()+i, nums.end());
			return;
		}
	}
	reverse(begin(nums), end(nums));
}