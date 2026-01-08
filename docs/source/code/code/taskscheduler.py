""" WHEN TASKS HAS TO BE EXECUTED IN ANY ORDER """
def taskScheduler(self, tasks: List[str], n: int) -> int:
	def heap_queue():
		# greedy:
		# since the order does not matter, pick the those teasks
		# in the decreasing order of counts.
		# 
		# IMPORTANT
		# process tasks with one tick at a time
		# to do that, we can maintain two data structures
		# (1) for the tasks which can be processed right away
		# (2) tasks which are in the queue to be available for processing later
		counts = Counter(tasks)
		# we don't need to reduce counts in the original dict
		# we can totally maintain the states using the two data structures we have
		# since we don't are about the type of the task, we don't need to keep
		# the task name as well
		# ready: priority queue -> counts
		# queued: counts, start_time
		ready = [-count for count in counts.values()]
		heapq.heapify(ready)
		queued = deque()
		curr_time = 0
		while ready or queued:
			if queued and queued[0][1] < curr_time:
				count, _ = queued.popleft()
				heapq.heappush(ready, count)
			if ready:
				count = 1 + heapq.heappop(ready)
				if count:
					queued.append((count, curr_time + n))
			curr_time += 1
		return curr_time

	def idle_counting():
		# visualisation:
		# 
	
	return heap_queue()

""" WHEN TASKS HAS TO BE EXECUTED IN GIVEN ORDER """
def taskSchedulerII(self, tasks: List[int], space: int) -> int:
	available_from = {}
	curr_day = 0
	for task in tasks:
		if task in available_from:
			""" IMPORTANT """
			# while doing timejump remember that we don't go back in past
			curr_day = max(curr_day, available_from[task]) # time jump
		curr_day += 1 # execute task
		available_from[task] = curr_day + space
	return curr_day
