string pushDominoes(string dominoes) {
	vector<int> distL(dominoes.size(), dominoes.size());
	vector<int> distR(dominoes.size(), dominoes.size());
	
	// distance to the closest 'L' on the right, until reset by a counter-balancing 'R'
	int pos = -1;
	for (int i = dominoes.size()-1; i >= 0; --i)
	{
		if (dominoes[i] == 'L')
			pos = i;
		else if (dominoes[i] == 'R')
			pos = -1;            
		if (pos != -1)
			distL[i] = pos - i;
	}
	
	// distance to the closest 'R' on the left, until reset by a counter-balancing 'L'
	pos = -1;
	for (int i = 0; i < dominoes.size(); ++i)
	{
		if (dominoes[i] == 'R')
			pos = i;
		else if (dominoes[i] == 'L')
			pos = -1;
		if (pos != -1)
			distR[i] = i - pos;
	}
	
	string s;
	s.resize(dominoes.size());
	
	for (int i = 0; i < dominoes.size(); ++i)
	{
		if (distL[i] < distR[i])
			s[i] = 'L';
		else if (distL[i] > distR[i])
			s[i] = 'R';
		else s[i] = dominoes[i];
	}
	
	return s;
}