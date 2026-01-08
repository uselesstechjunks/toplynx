vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph, int source, int target)
{
	auto res = dfs(graph, source, target);
	for (auto& curr : res)
	{
		std::reverse(curr.begin(), curr.end());
	}
	return res;
}

vector<vector<int>> dfs(vector<vector<int>>& graph, int u, int target)
{
	vector<vector<int>> res;
	bool found = process_vertex_early(u, target, res);
	if (!found)
	{
		// since it's DAG, discovered/processed flags are not required
		for (int v : graph[u])
		{
			auto curr = dfs(graph, v, target);
			// enter following loop only when target was found
			// this is ensured by the size of the return vector
			for (auto& v : curr)
			{
				// for each of the paths, append current node
				v.push_back(u);
				res.push_back(v);
			}
		}
	}
	return res;
}

bool process_vertex_early(int u, int target, vector<vector<int>>& res)
{
	if (u == target)
	{
		vector<int> curr(1);
		curr[0] = u;
		res.push_back(curr);
		return true;
	}
	return false;
}