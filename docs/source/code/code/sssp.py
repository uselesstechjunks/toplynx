# find the shortest path from k to all the nodes in the graph and then return the max
def relax(dist, u, v, w):
	if dist[v] > dist[u] + w:
		dist[v] = dist[u] + w
		return True
	return False

def bellman():
	dist = [inf] * (n + 1)
	dist[k] = 0
	# iterate as many times as there are number of edges
	for _ in range(n-1):
		# in every iteration, pick the minimum edge, relax the incident node
		for u, v, w in sorted(times, key=lambda x:x[2]):
			relax(dist, u, v, w)
	max_dist = max(dist[1:])
	return max_dist if max_dist < inf else -1

def build(n, times):
	adj = [[] for _ in range(n + 1)]
	for u, v, w in times:
		adj[u].append((v, w))
	return adj

def dijkstra():
	dist = [inf] * (n + 1)
	dist[k] = 0

	adj = build(n, times)
	# we explore each vertex
	# minheap should be ordered based on distance
	# in every iteration, we take the tail vertex which has the minimum distance
	# we relax all head vertices which are connected by that vertex
	# heap should contain the distance and the tail vertex
	# note that heap might contain distance values which are no longer valid due to relaxation
	# we should remove those
	minheap = [(0, k)]
	while minheap:
		prev_dist, u = heapq.heappop(minheap)
		if prev_dist != dist[u]:
			continue
		for v, w in adj[u]:
			relaxed = relax(dist, u, v, w)
			if relaxed:
				heapq.heappush(minheap, (dist[v], v))

	max_dist = max(dist[1:])
	return max_dist if max_dist < inf else -1
