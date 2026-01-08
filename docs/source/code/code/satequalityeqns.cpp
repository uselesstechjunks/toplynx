bool equationsPossible(vector<string>& equations) {
	vector<int> parents(26);
	iota(parents.begin(), parents.end(), 0);
	
	for (string eqn : equations)
	{
		int a = eqn[0]-'a';
		int b = eqn[3]-'a';
		
		if (eqn[1] == '=')
		{
			parents[find(parents, a)] = find(parents, b);
		}
	}
	
	for (string eqn : equations)
	{
		int a = eqn[0]-'a';
		int b = eqn[3]-'a';
		
		if (eqn[1] == '!')
		{
			if (find(parents, a) == find(parents, b))
				return false;
		}
	}
	
	return true;
}

int find(const vector<int>& parents, int x)
{
	if (parents[x] == x)
		return x;
	return find(parents, parents[x]);
}