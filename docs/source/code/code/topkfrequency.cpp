vector<string> topKFrequent(vector<string>& words, int k) {
	unordered_map<string,int> freq;
	for (int i = 0; i < words.size(); ++i)
	{
		if (freq.find(words[i]) == freq.end())
			freq.insert({words[i], 1});
		else
			freq[words[i]]++;
	}
	// min heap:: uses a > b comparator
	auto cmp = [&freq](const string& a, const string& b)
	{
		return freq[a] > freq[b] || (freq[a] == freq[b] && a < b);
	};
	priority_queue<string,vector<string>,decltype(cmp)> heap(cmp);
	for (auto item : freq)
	{
		heap.push(item.first);
		if (heap.size() > k)
		{
			heap.pop();
		}
	}
	vector<string> res(k);
	int i = k-1;
	while (!heap.empty())
	{
		res[i--] = heap.top();
		heap.pop();
	}
	return res;
}