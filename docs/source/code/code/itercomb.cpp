class CombinationIterator {
public:
	CombinationIterator(string characters, int combinationLength) {
		curr = 0;
		length = combinationLength;
		string buf(combinationLength, 0);
		backtrack(characters, buf, 0, 0);
	}

	string next() {
		return combinations[curr++];
	}

	bool hasNext() {
		return curr < combinations.size();
	}
private:
	// k: index of the result string
	// j: index of the letter in the sorted characters string to pick from
	void backtrack(const string& characters, string& buf, int k, int j)
	{
		if (k == length)
		{
			combinations.push_back(buf);
			return;
		}

		// we've already used all the letters in sorted order before 'j'
		// therefore, to form the result, we loop through all the possible options from 'j'
		for (int i = j; i < characters.size(); ++i)
		{
			buf[k] = characters[i];
			// !!IMPORTANT!!
			// since we just used 'i'-th letter, make sure to pass that info correctly
			// in the next call, the first letter to pick from would be i+1, NOT j+1
			backtrack(characters, buf, k+1, i+1);
		}
	}
	vector<string> combinations;
	size_t curr;
	int length;
};

/**
* Your CombinationIterator object will be instantiated and called as such:
* CombinationIterator* obj = new CombinationIterator(characters, combinationLength);
* string param_1 = obj->next();
* bool param_2 = obj->hasNext();
*/