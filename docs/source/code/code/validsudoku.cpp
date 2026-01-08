bool isValidSudoku(vector<vector<char>>& board) {
    vector<unordered_set<int>> rows(9);
    vector<unordered_set<int>> cols(9);
    vector<unordered_set<int>> grids(9);

    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            if (board[i][j] == '.')
                continue;

            int curr = board[i][j]-'1';

            if (rows[curr].find(i) != rows[curr].end())
                return false;
            else
                rows[curr].insert(i);

            if (cols[curr].find(j) != cols[curr].end())
                return false;
            else
                cols[curr].insert(j);

            if (grids[curr].find(i/3*3+j/3) != grids[curr].end())
                return false;
            else
                grids[curr].insert(i/3*3+j/3);
        }
    }

    return true;
}
