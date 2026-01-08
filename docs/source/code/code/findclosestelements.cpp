vector<int> findClosestElements(vector<int>& arr, int k, int x) {
	auto cmp = [&x](int a, int b)
	{
		return abs(a-x) < abs(b-x) || (abs(a-x) == abs(b-x) && a < b);
	};
	priority_queue<int,vector<int>,decltype(cmp)> heap(cmp);
	for (int i = 0; i < arr.size(); ++i)
	{
		heap.push(arr[i]);
		if (heap.size() > k)
			heap.pop();
	}
	vector<int> res(k, -1);
	int i = 0;
	while (!heap.empty())
	{
		res[i++] = heap.top();
		heap.pop();
	}
	sort(res.begin(), res.end());
	return res;
}