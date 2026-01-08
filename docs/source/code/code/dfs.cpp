void dfs(vector<vector<int>>& graph, bool directed)
{
	vector<state> status(graph.size(), UNDISCOVERED);
	vector<int> parents(graph.size(), -1);
	vector<int> entry(graph.size(), -1);
	vector<int> exit(graph.size(), -1);
	int time = 0;
	for (int u = 0; u < graph.size(); ++u)
	{
		if (state[u] != DISCOVERED)
		{
			 dfs(graph, u, parents, entry, exit, time, directed);
		}
	}
}

void dfs(vector<vector<int>>& graph,
		 int u,
		 vector<int>& parents,
		 vector<int>& entry,
		 vector<int>& exit,
		 int& time,
		 bool directed)
{
	status[u] = DISCOVERED;
	time = time + 1;
	entry[u] = time;
	process_vertex_early(u);
		
	for (int v : graph[u])
	{
		if (status[v] != DISCOVERED)
		{
			// v is undiscovered => (u,v) is a tree-edge
			status[v] = DISCOVERED;
			parent[v] = u;
			process_edge(u, v);
			dfs(graph, v, parents, entry, exit, time, directed);
		}
		else if ((status[v] != PROCESSED && parents[u] != v) || directed)
		{
			// v is discovered but not processed state => v is an ancestor
			// if v is an immediate ancestor, this is a second visit to the edge (v <-> u)
			// but if v is distant ancestor, then this is a back-edge
			process_edge(u, v);
		}
	}
		
	process_vertex_late(u);
	time = time + 1;
	exit[u] = time;
	status[u] = PROCESSED;
}