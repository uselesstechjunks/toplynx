def minMeetingRooms(self, intervals: List[List[int]]) -> int:
	n = len(intervals)
	intervals.sort()
	available = []
	# rooms start from id 1
	occupied = [] # stores the finish time and room id
	req_rooms = 0 

	for interval in intervals:
		# when there are meetings which ends before the current
		# meeting starts, we remove them from occupied and mark
		# them as available
		while occupied and occupied[0][0] <= interval[0]:
			finish_time, room = heapq.heappop(occupied)
			heapq.heappush(available, room)
		if available:
			room = heapq.heappop(available)
			heapq.heappush(occupied, (interval[1], room))
		else:
			req_rooms += 1
			heapq.heappush(occupied, (interval[1], req_rooms))
	return req_rooms