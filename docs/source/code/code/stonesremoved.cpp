int removeStones(vector<vector<int>>& stones)
{
	vector<vector<int>> graph(stones.size());
	for (int i = 0; i < stones.size(); ++i)
	{
		// no self-loop by design
		for (int j = i+1; j < stones.size(); ++j)
		{
			if (stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1])
			{
				// undirected by design
				graph[i].push_back(j);
				graph[j].push_back(i);
			}
		}
	}

	int components = 0;
	vector<bool> visited(graph.size(), false);
	for (int i = 0; i < graph.size(); ++i)
	{
		if (!visited[i])
		{
			dfs(graph, visited, i);
			++components;
		}
	}

	return stones.size() - components;
}

void dfs(const vector<vector<int>>& graph, vector<bool>& visited, int i)
{
	if (visited[i])
		return;

	visited[i] = true;
	for (int j = 0; j < graph[i].size(); ++j)
	{
		dfs(graph, visited, graph[i][j]);
	}
}