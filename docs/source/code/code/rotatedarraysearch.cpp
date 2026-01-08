int findPivotElement(vector<int>& nums)
{
	int l = 0;
	int r = nums.size() - 1;
	while (nums[l] > nums[r])
	{
		int m = l + (r - l) / 2;
		if (nums[l] > nums[m])
			r = m;
		else if (nums[m] > nums[r])
			l = m + 1;
	}
	return l;
}

int search(vector<int>& nums, int target) {
	int offset = findPivotElement(nums);
	int l = 0;
	int r = nums.size()-1;
	while (l <= r)
	{
		int m = l + (r - l) / 2;
		int real_m = (m + offset) % nums.size();
		if (nums[real_m] == target)
			return real_m;
		else if (nums[real_m] < target)
			l = m + 1;
		else
			r = m - 1;
	}
	return -1;
}