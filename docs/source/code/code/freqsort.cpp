string frequencySort(string s) 
{
    unordered_map<char,int> counts;
    for (char c : s)
    {
        if (counts.find(c) != counts.end())
        {
            counts.insert({c,0});
        }
        counts[c]++;
    }

    auto cmp = [&counts](char a, char b)
    {
        return counts[a] < counts[b];
    };

    priority_queue<char,vector<char>,decltype(cmp)> heap(cmp);
    for (auto kv : counts)
    {
        heap.push(kv.first);
    }

    int i = 0;
    while (!heap.empty())
    {
        char c = heap.top();
        heap.pop();
        for (int j = 0; j < counts[c]; ++j)
        {
            s[i++] = c;
        }
    }

    return s;
}
