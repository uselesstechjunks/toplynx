vector<vector<int>> findWinners(vector<vector<int>>& matches) {
    set<int> players;
    unordered_map<int,int> losers;

    for (int i = 0; i < matches.size(); ++i)
    {
        int winner = matches[i][0];
        int loser = matches[i][1];

        if (players.find(winner) == players.end())
        {
            players.insert(winner);
        }

        if (players.find(loser) == players.end())
        {
            players.insert(loser);
        }

        if (losers.find(loser) == losers.end())
        {
            losers.insert({loser, 0});
        }
        losers[loser]++;
    }
    vector<vector<int>> result(2);
    for (int player : players)
    {
        if (losers.find(player) == losers.end())
        {
            result[0].push_back(player);
        }
        else if (losers[player] == 1)
        {
            result[1].push_back(player);
        }
    }
    return result;
}
