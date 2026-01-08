vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
	vector<bool> aux(1000, false);
	vector<int> ans(std::min(nums1.size(), nums2.size()));
	for (int i = 0; i < nums1.size(); ++i)
	{
		aux[nums1[i]] = true;
	}
	int count = 0;
	for (int i = 0; i < nums2.size(); ++i)
	{
		if (aux[nums2[i]])
		{
			ans[count] = nums2[i];
			aux[nums2[i]] = false;
			++count;
		}
	}
	ans.resize(count);
	return ans;
}