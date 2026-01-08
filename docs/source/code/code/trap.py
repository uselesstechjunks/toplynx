def trap(height: List[int]) -> int:
	"""
	Notation: Let m(i,j) is max(height[i,j]) both ends included.
	Observation: for every bar at i, keep track of 2 quantities.
	(1) max on the left including i, m(0,i) 
	(2) max on the right including i, m(i,n-1)
	Case #1: h(i) is smaller than both
		water stored at the top of i would be determined by whichever is smaller
		between these two m(0,i) and m(i,n-1).
			ontop(i) = min(m(0,i),m(i,n-1)) - h(i)
	Case #2: h(i) itself is either m(0,i) or m(i,n-1)
		water stored in top of i would be 0 in this case. note that the previous
		expression still applies here and evaluates to 0.
			ontop(i) = min(m(0,i),m(i,n-1)) - h(i)
	Note: There are no other cases possible.
	If we keep track of m(0,i) and m(i,n-1) in 2 separate arrays, the solution is simple.

		n = len(height)
		maxleft, maxright = height[:], height[:]
		for i in range(1,n):
			maxleft[i] = max(maxleft[i-1], maxleft[i])
		for i in range(n-2,-1,-1):
			maxright[i] = max(maxright[i], maxright[i+1])
		sum = 0
		for i in range(n):
			sum += min(maxleft[i], maxright[i]) - height[i]
		return sum
	
	We can do better. The key is elimination of choice using ordering property.

	Note that at each step, we only need min(m(0,i), m(i,n-1)).
	
	Let's say, we haven't explored the right side completely, just explored height[j,n-1]
	and have m(j,n-1) with us, and m(j,n-1) is already larger than m(0,i).
	
	It is totally possible that if we investigate the rest of the right side, we might
	find something that's larger than m(j,n-1), i.e. m(i,n-1) > m(j,n-1).
	
	But knowing that wouldn't change the fact that min(m(0,i),m(i,n-1)) = m(0,i) in this case.
	The water on top would always be determined by the smaller one, i.e. m(0,i).

	So, if we keep track of max on both ends, m(0,i) and m(j,n-1), that would suffice
	to determine water on top deterministically.

	If m(0,i) < m(j,n-1), we already know water on top of i. We add and move on.
	If m(0,i) > m(j,n-1), we already know water on top of j. We add and move on.
	"""
	n = len(height)
	i, j = 0, n-1
	maxleft, maxright = height[i], height[j]
	total = 0
	while i < j:
		maxleft = max(maxleft, height[i])
		maxright = max(maxright, height[j])
		if maxleft < maxright:
			total += maxleft - height[i]
			i += 1
		else:
			total += maxright - height[j]
			j -= 1
	return total