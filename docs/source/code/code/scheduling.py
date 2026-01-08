def canAttendAllMeetings(self, intervals: List[List[int]]) -> bool:
	def overlaps(first, second):
		return first[1] > second[0]

	n = len(intervals)
	# sort the intervals by start time
	# sorting helps because if any given interval (curr)
	# doesn't overlaps with the immediate next interval
	# then none of the next intervals would be overlapping with curr
	intervals.sort(key=lambda x:x[0])
	for i in range(1, n):
		if overlaps(intervals[i-1], intervals[i]):
			return False
	return True

def maxTimeMeetingRoomBooked(self, intervals: List[List[int]]) -> List[List[int]]:
	def overlaps(interval1, interval2):
		# since the calling ensures that interval1 has earlier start time
		# than interval2, we just need to check if interval1 ends after
		# interval2 begins.
		return interval1[1] >= interval2[0]
	
	def merge(interval1, interval2):
		# similar to above, the calling ensures that interval1 starts earlier
		# than interval2. therefore, for merged interval, we only need to check
		# the max of end times
		return [interval1[0], max(interval1[1], interval2[1])]

	intervals.sort()
	stack = [intervals[0]]

	for interval in intervals[1:]:
		if overlaps(stack[-1], interval):
			last = stack.pop()
			stack.append(merge(last, interval))
		else:
			stack.append(interval)

	return stack

def minMeetingsDropped(self, intervals: List[List[int]]) -> int:
	# removing overlapping intervals is a different prolem
	# than counting intervals.
	# why?
	# say, we have 3 intervals, i1, i2, i3 where i1 overlaps with i2
	# and i2 overlaps with i3. if we count overlaps, there are 2.
	# if we remove i1, overlap count reduces to 1
	# if we remove i3, overlap count reduces to 1
	# but if we remove i2, overlap count reduces to 0
	# so this requires a different strategy
	# greedy activity selection. basically the idea is to think of it
	# as meetings which are to be held in a single room. we are the doorman
	# and we want to maximize the room usage by assigning it to the most
	# number of meetings possible. this can be achieved by assigning it to
	# the meeting that ends first. since this choice doesn't hurt with the objective.
	intervals.sort(key=lambda x:x[1]) # sorted by end time
	n = len(intervals)
	meetings_count = 1
	last_meeting_finish_time = intervals[0][1]
	for start_time, finish_time in intervals[1:]:
		if last_meeting_finish_time <= start_time:
			meetings_count += 1
			last_meeting_finish_time = finish_time
	return n - meetings_count

def minMeetingRooms(self, intervals: List[List[int]]) -> int:
	# it asks for minimum number of resources that is required
	# to allocate all meetings. this means, every time there
	# are overlapping meetings, we need to find a new room to 
	# accomodate for it. if intervals are represented as nodes in
	# a graph, and we use edges to connect those nodes whenever they
	# are overlapping, then this problem can be thought of as a
	# graph colouring problem on that interval graph.
	# the algorithm works by simulation of the situation.
	# we need to maintain two lists - one for occupied rooms and one
	# for free rooms that we've already decided to pay for.
	# every time a new meeting comes, we first check if there are occupied
	# rooms which can be marked as free. we add them to the free rooms list.
	# we allocate room from this free list as we've already paid for them.
	# if there really are no free rooms, we increase max room count, and decide
	# to pay for one more extra room
	intervals.sort()
	occupied = [] # needs to store finish time
	# instead of free list, can we manage using just the count?
	free_rooms_count = 1
	max_rooms_used = 1

	for start_time, finish_time in intervals:
		while occupied and occupied[0] <= start_time:
			heapq.heappop(occupied)
			free_rooms_count += 1
		if free_rooms_count:
			heapq.heappush(occupied, finish_time)
			free_rooms_count -= 1
		else:
			max_rooms_used += 1
			heapq.heappush(occupied, finish_time)
	
	return max_rooms_used

def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
	# here, the room id matters
	# need to maintain not just free rooms, but the room ids as well
	# lucky for us, our allocation rule indicates that we can pick
	# rooms from the free list in order of their ids.
	# so we maintain the same occupied and free lists. but we also
	# store the room ids.
	meetings.sort()
	occupied = [] # stores finish time and room id
	free = list(range(n)) # stores room id

	# we also need to do bookkeeping for meetings counts
	counts = defaultdict(int)

	for start_time, end_time in meetings:
		duration = end_time - start_time

		while occupied and occupied[0][0] <= start_time:
			_, room_id = heapq.heappop(occupied)
			heapq.heappush(free, room_id)
		
		if free:
			# can allocate a room right now
			room_id = heapq.heappop(free)
			heapq.heappush(occupied, (end_time, room_id))
			counts[room_id] += 1
		else:
			# need to wait until another room becomes free
			# need to adjust end time accordingly
			finish_time, room_id = heapq.heappop(occupied)
			heapq.heappush(occupied, (finish_time + duration, room_id))
			counts[room_id] += 1

	max_meetings = max(counts.values())
	# we return the room that held max meetings and has the lowest index
	# this can be achieved by traversing the key list in counts in sorted
	# order. lucky for us, dict already has keys in sorted order
	return [x for x in counts if counts[x] == max_meetings][0]

def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
	# similar to meeting roms 3, here also we need to maintain
	# two lists for free resources and occupied resources
	# but unlike meeting rooms 3, where we were to wait until a room was free
	# and our preference was to allocate rooms to the lowest id possible,
	# here, we have a fixed allocation rule according to round robin.
	# if there are no free rooms at the time of arrival, request is dropped.
	# the round robin works on sorted room id sequences.
	# example:
	# at any given time, free rooms [0,1,5], and we have 8 resources [0-7]
	# request 12 comes. 12 % 8 = 4
	# if server 4 was available, we would have used it. but since it's not,
	# the next best we can do is 5. note that this is the insert position of 4
	# in the sorted free rooms list
	# does this hold for everything?
	# any incoming requests would be mapped to [0, 7] due to modulo division.
	# say, request 16 comes. server index = 16 % 8 = 0
	# since 0 is already there, we allocate it.

	# arrivals are already sorted
	occupied = [] # stores the finish time and room_id
	free = SortedList(list(range(k))) # stores room ids in sorted order
	# SortedList is a balanced BST
	# we could have used SortedSet since rooms are unique
	# but SortedList provides us with bisect_left which is what we need

	# bookkeeping for counts
	counts = defaultdict(int)

	for i, start_time in enumerate(arrival):
		end_time = start_time + load[i]

		while occupied and occupied[0][0] <= start_time:
			_, server_id = heapq.heappop(occupied)
			free.add(server_id)
		
		if free:
			# key point to observe here is that as long as we
			# have free servers, we can ALWAYS allocate resource
			# in a round robin fashion
			server_id = i % k
			index = free.bisect_left(server_id)
			# there is a server which can serve the request
			server_id = free[index] if index < len(free) else free[0]
			free.remove(server_id)
			heapq.heappush(occupied, (end_time, server_id))
			counts[server_id] += 1
	
	max_count = max(counts.values())
	# since keys in dict are in sorted order, this returns
	# the list of server ids in sorted order
	return [x for x in counts if counts[x] == max_count]
