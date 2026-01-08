void bfs(vector<vector<int>>& graph, int start, bool directed)
{
	queue<int> q;
	// state: UNDISCOVERED, DISCOVERED, PROCESSED
	vector<state> status(graph.size(),UNDISCOVERED);
	vector<int> parents(graph.size(),-1);
	
	status[start] = DISCOVERED;
	q.push(start);
	
	while (!q.empty())
	{
		int u = q.front();
		q.pop();
		process_vertex_early(u);
		
		for (int v : graph[u])
		{
			if (status[v] != DISCOVERED)
			{
				status[v] = DISCOVERED;
				q.push(v);
				parent[v] = u;
			}
			if (status[v] != PROCESSED || directed)
			{
				process_edge(u, v);
			}
		}
		
		process_vertex_late(u)
		status[u] = PROCESSED;
	}
}