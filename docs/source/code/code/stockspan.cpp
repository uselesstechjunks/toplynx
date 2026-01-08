class StockSpanner {
public:
	StockSpanner() {
		last_index = 0;
	}

	int next(int price) {        
		int span = last_index + 1;
		while (!st.empty() && st.top().second <= price)
		{
			st.pop();
		}
		if (!st.empty())
		{
			span = last_index - st.top().first;
		}
		st.push({last_index, price});
		++last_index;
		return span;
	}
	int last_index;
	stack<pair<int,int>> st;
};