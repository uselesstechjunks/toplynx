int findKthLargest(vector<int>& nums, int k) {
	priority_queue<int,vector<int>,std::greater<int>> heap;
	for (int i = 0; i < nums.size(); ++i)
	{
		heap.push(nums[i]);
		if (heap.size() > k)
			heap.pop();
	}
	return heap.top();
}