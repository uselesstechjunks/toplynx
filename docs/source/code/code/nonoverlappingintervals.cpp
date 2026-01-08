int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    auto cmp = [](const vector<int>& i1, const vector<int>& i2)
    {
        if (i1[1] == i2[1])
        {
            return i1[0] < i2[0];
        }
        return i1[1] < i2[1];
    };

    sort(intervals.begin(), intervals.end(), cmp);
    pair<int,int> last = {intervals[0][0], intervals[0][1]};
    int overlapCount = 0;

    for (size_t i = 1; i < intervals.size(); ++i)
    {
        pair<int,int> curr = {intervals[i][0], intervals[i][1]};

        if (last.second > curr.first)
        {
            ++overlapCount;
        }
        else
        {
            last = curr;
        }
    }

    return overlapCount;
}
