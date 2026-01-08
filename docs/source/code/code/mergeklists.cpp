ListNode* mergeKLists(vector<ListNode*>& lists) {
	auto cmp = [](ListNode* n1, ListNode* n2)
	{
		return n1->val > n2->val;
	};
	priority_queue<ListNode*,vector<ListNode*>,decltype(cmp)> q(cmp);
	for (int i = 0; i < lists.size(); ++i)
	{
		if (lists[i] != nullptr)
			q.push(lists[i]);
	}
	ListNode* head = nullptr;
	ListNode* last = nullptr;
	while (!q.empty())
	{
		ListNode* node = q.top();
		q.pop();
		if (node->next != nullptr)
			q.push(node->next);
		if (head == nullptr)
		{
			head = node;
			last = node;
		}
		else
		{
			last->next = node;
			last = node;
		}
	}
	return head;
}