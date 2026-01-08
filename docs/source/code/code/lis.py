def use_patience_solitaire():
	# we take the numbers as cards and deal them one by one to create decks
	# (a) for a card, check the decks from left to right
	# (b) if the card is smaller than the card on that deck, it goes on top
	# (c) else we create a new deck with the card
	# Note:
	# (a) if we form the decks this way, the within a given deck, the numbers are
	#     arranged in decreasing order
	# (b) the top card of the decks would be in increasing order from left to right
	# (c) the number of decks created this way is the length of longest increasing subsequence
	decks = [nums[0]]
	for num in nums[1:]:
		if decks and decks[-1] < num:
			decks.append(num)
		else:
			insert_index = bisect_left(decks, num)
			decks[insert_index] = num
	return len(decks)

def use_dp():
	# lis[i] is the longest increasing subsequence ending at nums[i]
	# once I see nums[i+1], I need to consider all previous lis[k]
	# instances k <= i, where nums[k] < nums[i+1]. The new number is
	# able to create an lis of an increased length by 1
	# base case: every lis[i] is at least 1
	n = len(nums)
	dp = [1] * n
	max_len = 1
	for i in range(1, n):
		for k in range(i):
			if nums[k] < nums[i]:
				dp[i] = max(dp[i], dp[k] + 1)
				max_len = max(max_len, dp[i])
	return max_len
