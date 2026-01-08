ListNode* oddEvenList(ListNode* head) 
{
    if (head == nullptr || head->next == nullptr)
        return head;

    ListNode* evenHead = head->next;
    ListNode* odd = head;
    ListNode* even = head->next;

    while (even != nullptr && even->next != nullptr)
    {
        ListNode* oddNext = even->next;
        ListNode* evenNext = oddNext->next;
        odd->next = oddNext;
        even->next = evenNext;
        odd = odd->next;
        even = even->next;
    }

    odd->next = evenHead;
    return head;
}
