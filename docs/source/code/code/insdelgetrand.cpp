class RandomizedSet {
public:
    RandomizedSet() {
        
    }
    
    bool insert(int val) {
        if (map.find(val) != map.end())
        {
            return false;
        }
        map.insert({val,nums.size()});
        nums.push_back(val);
        return true;
    }
    
    bool remove(int val) {
        if (map.find(val) == map.end())
        {
            return false;
        }
        // swap the deleted val with the last val in num
        int lastval = nums[nums.size()-1];
        nums[map[val]] = lastval;
        map[lastval] = map[val];
        map.erase(val);
        nums.pop_back();
        return true;
    }
    
    int getRandom() {
        return nums[rand() % nums.size()];
    }
private:
    vector<int> nums;
    unordered_map<int,int> map;
};

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet* obj = new RandomizedSet();
 * bool param_1 = obj->insert(val);
 * bool param_2 = obj->remove(val);
 * int param_3 = obj->getRandom();
 */
