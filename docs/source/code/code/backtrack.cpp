// call with backtrack(a, 0, input);
// input can contain anything - e.g. the stopping condition (size)
void backtrack(vector<int>& a, int k, data input)
{
	if (isSolution(a, k, input))
		processSolution(a, k, input);
	else
	{
		k = k + 1;
		vector<int> candidates = generateCandidates(a, k, input);
		for (auto candidate : candidates)
		{
			a[k] = candidate;
			make_move(a, k, input);
			backtrack(a, k, input);
			unmake_move(a, k, input);
		}
	}
}