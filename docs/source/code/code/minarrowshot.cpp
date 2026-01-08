int findMinArrowShots(vector<vector<int>>& points) {
    auto cmp = [](const vector<int>& p1, const vector<int>& p2)
    {
        if (p1[1] == p2[1])
        {
            return p1[0] < p2[0];
        }
        return p1[1] < p2[1];
    };

    sort(points.begin(), points.end(), cmp);
    pair<int,int> shotRange = {points[0][0], points[0][1]};
    int count = 1;

    for (size_t i = 1; i < points.size(); ++i)
    {
        pair<int,int> curr = {points[i][0], points[i][1]};
        if (shotRange.second >= curr.first)
        {
            shotRange.first = curr.first;
        }
        else
        {
            ++count;
            shotRange = curr;
        }
    }

    return count;
}
