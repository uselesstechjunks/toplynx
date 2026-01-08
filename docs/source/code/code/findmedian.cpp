class MedianFinder {
public:
	MedianFinder() {

	}

	void addNum(int num) {
		// special case when max_heap is empty
		if (max_heap.empty())
		{
			max_heap.push(num);
			return;
		}

		// size maintanining
		if (max_heap.size() > min_heap.size())
		{
			min_heap.push(num);
		}
		else
		{
			max_heap.push(num);
		}

		// left_max and right_min invariance maintaining
		if (max_heap.top() > min_heap.top())
		{
			int left_max = max_heap.top();
			int right_min = min_heap.top();
			max_heap.pop();
			min_heap.pop();
			max_heap.push(right_min);
			min_heap.push(left_max);
		}
	}

	double findMedian() {
		if (max_heap.size() > min_heap.size())
			return max_heap.top();
		return (max_heap.top() + min_heap.top()) / 2.0;
	}
private:
	priority_queue<int,vector<int>,greater<int>> min_heap;
	priority_queue<int,vector<int>,less<int>> max_heap;
};

/**
* Your MedianFinder object will be instantiated and called as such:
* MedianFinder* obj = new MedianFinder();
* obj->addNum(num);
* double param_2 = obj->findMedian();
*/