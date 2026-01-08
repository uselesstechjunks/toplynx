vector<int> productExceptSelf(vector<int>& nums) {
	vector<int> res(nums.size(), 0);
	int num_zeros = 0;
	int zero_index = -1;
	int product = 1;
	int product_non_zeros = 1;
	for (int i = 0; i < nums.size(); ++i)
	{
		product *= nums[i];
		if (nums[i] == 0)
		{
			++num_zeros;
			zero_index = i;
		}
		else
		{
			product_non_zeros *= nums[i];
		}
	}
	
	if (num_zeros > 1)
		return res;
	
	if (num_zeros == 1)
	{
		res[zero_index] = product_non_zeros;
		return res;
	}
	
	for (int i = 0; i < nums.size(); ++i)
	{
		res[i] = product / nums[i];
	}
	
	return res;
}