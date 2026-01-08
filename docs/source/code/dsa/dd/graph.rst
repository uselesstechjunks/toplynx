================================================================================
Advanced Graph Topics
================================================================================
.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

1. Shortest Paths
--------------------------------------------------------------------------------
* Sample implementation

	.. collapse:: Single source shortest path

	   .. literalinclude:: ../../code/sssp.py
	      :language: python
	      :tab-width: 2
	      :linenos:

* Why important: Many problems in real-world applications (e.g., routing, network optimization) rely on shortest paths.
* Relevant Algorithms:

	* Dijkstra’s Algorithm (for non-negative weights).
	* Bellman-Ford Algorithm (for graphs with negative weights).
	* Floyd-Warshall Algorithm (all-pairs shortest paths).
	* A* Search (if heuristic-based optimization is required).
* Example Problems:

	* Find the shortest path in a weighted directed graph.
	* Determine if a negative weight cycle exists.
	* Optimize routing in a graph with mixed positive and negative weights.
* More Problems:

	#. Problem: Given a weighted directed graph, find the shortest path from a source to a destination with at most :math:`k` intermediate nodes.  
	#. Similar to the above, but you must use exactly :math:`k` edges.
	#. Detect if there's a negative weight cycle and compute the shortest path in its presence (if possible).	
	#. Find the shortest path where certain nodes must or must not be visited.
	#. Each edge can only be traversed a fixed number of times (e.g., roads with toll limits).
	#. Find the shortest path where one edge’s weight can be reduced by a discount or percentage.
	#. Edge weights vary based on the time of traversal (e.g., traffic at different times).
	#. Find the shortest path that visits all nodes at least once.
	#. Edge weights change dynamically during traversal (e.g., based on usage or traffic updates).	
	#. Compute the shortest path between multiple source-destination pairs.
	#. Find the shortest path in a grid where some cells are blocked, and you can break at most :math:`k` obstacles.
	#. Different edge weights for different "modes" (e.g., car, bike, foot).

2. Minimum Spanning Tree (MST)
--------------------------------------------------------------------------------
* Sample implementation

	.. collapse:: MST algorithms

	   .. literalinclude:: ../../code/mst.py
	      :language: python
	      :tab-width: 2
	      :linenos:

* Why important: MSTs are useful in optimization problems, especially those involving connectivity.
* Key Algorithms:

	* Prim’s Algorithm.
	* Kruskal’s Algorithm.
* Example Problems:

	* Compute the MST for a weighted undirected graph.
	* Update the MST dynamically when a new edge is added.
	* Determine the second-best MST.
* More Problems:

	#. Find a minimum spanning tree, but the tree must include a specific edge :math:`(u, v)`.
	#. Find a spanning tree with the maximum total weight instead of the minimum.
	#. Find the second smallest weight of a spanning tree (not the same as the minimum spanning tree).
	#. Find a minimum spanning tree where no vertex can have a degree greater than `k`.
	#. Each edge has an associated penalty if it is not included in the spanning tree. Find the tree that minimizes the sum of the MST weight and the penalties of excluded edges.
	#. Find an MST where a specific vertex `v` must be part of the tree.
	#. Each edge is assigned a color, and the MST must include at least one edge of every color.
	#. You are given an MST for a graph. Process queries to either:
			
		- Add an edge and update the MST.
		- Remove an edge and update the MST.
	#. Some edges have a discounted weight (e.g., weight reduced by `x`). Find the MST under the discounted weights.
	#. Find the MST and also compute, for each edge in the MST, the cost of the MST if that edge is removed.
	#. Find an MST in a graph that includes edges with negative weights.
	#. Find an MST where the maximum depth of any vertex from the root is less than or equal to `k`.

3. Topological Sort
--------------------------------------------------------------------------------
* Sample implementation

	.. collapse:: Single source shortest path

	   .. literalinclude:: ../../code/tsort.py
	      :language: python
	      :tab-width: 2
	      :linenos:
* Why important: Crucial for dependency resolution and scheduling problems.
* Key Techniques:

	* Kahn’s Algorithm (BFS-based).
	* DFS with post-order traversal.
* Example Problems:

	* Check if a directed graph has a cycle.
	* Compute a valid topological ordering.
	* Find the number of valid topological orderings.

4. Strongly Connected Components (SCCs)
--------------------------------------------------------------------------------
* Why important: SCCs are foundational in analyzing directed graphs for connectivity.
* Key Algorithms:

	* Kosaraju’s Algorithm.
	* Tarjan’s Algorithm.
* Example Problems:

	* Find all SCCs in a directed graph.
	* Determine if a graph is strongly connected.
	* Compute the smallest set of edges to make a graph strongly connected.

5. Bipartite Graphs
--------------------------------------------------------------------------------
* Why important: Common in matching and coloring problems.
* Sample implementation

	.. collapse:: Using BFS and DFS

	   .. literalinclude:: ../../code/bipartite.py
	      :language: python
	      :tab-width: 2
	      :linenos:
* Key Techniques:

	* BFS/DFS to test bipartiteness.
	* Maximum Bipartite Matching using augmenting paths.
* Example Problems:

	* Check if a graph is bipartite.
	* Solve matching problems in bipartite graphs.
	* Partition the graph into two disjoint sets.

6. Graph Traversals
--------------------------------------------------------------------------------
* Sample implementation

	.. collapse:: A collection of traversal algorithms and applications

	   .. literalinclude:: ../../code/graph.py
	      :language: python
	      :tab-width: 2
	      :linenos:
* Bidirectional BFS

	.. collapse:: Word Ladder

	   .. literalinclude:: ../../code/wordladder.py
	      :language: python
	      :tab-width: 2
	      :linenos:
* Multi Source BFS

	.. collapse:: Rotting Oranges

	   .. literalinclude:: ../../code/rottingoranges.py
	      :language: python
	      :tab-width: 2
	      :linenos:
* Why important: Breadth-first and depth-first searches are foundational for exploring graphs.
* Key Techniques:

	* BFS (used for shortest paths in unweighted graphs, connected components).
	* DFS (used for cycle detection, pathfinding, and SCCs).
* Example Problems:

	* Find all connected components.
	* Detect cycles in a directed or undirected graph.
	* Implement BFS/DFS to solve maze problems.

7. Dynamic Graph Algorithms
--------------------------------------------------------------------------------
* Why important: Company values efficiency, and dynamic updates test your ability to optimize graph data structures.
* Key Problems:

	* Maintain connectivity as edges are added or removed.
	* Recompute shortest paths or MST dynamically.
	* Optimize graph updates in streaming contexts.

8. Network Flow
--------------------------------------------------------------------------------
* Why important: Advanced but occasionally tested for senior-level candidates to assess problem-solving depth.
* Key Algorithms:

	* Ford-Fulkerson Algorithm.
	* Edmonds-Karp Algorithm.
* Example Problems:

	* Compute maximum flow in a flow network.
	* Solve bipartite matching using flow techniques.
	* Minimize the cut in a weighted graph.

9. Eulerian and Hamiltonian Paths
--------------------------------------------------------------------------------
* Why important: Rare but can appear in challenging questions.
* Example Problems:

	* Determine if a graph has an Eulerian path or circuit.
	* Find the Hamiltonian path if it exists.
	* Compute a path visiting all edges or vertices exactly once.

10. Advanced Graph Techniques
--------------------------------------------------------------------------------
* Why important: Tests your depth of knowledge for senior-level positions.
* Key Areas:

	* Articulation Points and Bridges.
	* Graph Coloring Problems.
	* Spectral Graph Theory (rare but valuable for specific roles).
* More Problems:

	#. Determine the chromatic number of a graph, i.e., the minimum number of colors required to color the graph such that no two adjacent vertices share the same color.
	#. Check if a graph is bipartite by verifying if it can be colored using exactly two colors.	
	#. Assign colors to edges such that no two edges sharing the same vertex have the same color. Minimize the number of colors used.	
	#. Color a graph such that certain vertices have preassigned colors or cannot use specific colors.
	#. Assign colors such that no two vertices at a distance of 1 (adjacent) or distance of 2 (neighbors' neighbors) share the same color.
	#. Assign colors to vertices such that the sum of the weights of conflicting edges is minimized.
	#. Given a fixed number of colors, determine if the graph can be properly colored.
	#. Assign colors to vertices such that the difference between the colors of adjacent vertices satisfies specific modular constraints.	
	#. Color a planar graph with a maximum of 4 colors (Four Color Theorem).
	#. Maintain a valid coloring of a graph while allowing for vertex or edge insertions and deletions.

11. All Topics
--------------------------------------------------------------------------------
#. You are given a directed graph where each node represents a city and edges represent roads between them with a time cost. Find the smallest time to travel between two given cities, but you can use a "shortcut" road that reduces the time of any one edge to zero.
#. A maze is represented as a grid. Each cell is either walkable or a wall. Find the minimum number of walls you must break to create a path from the top-left corner to the bottom-right corner.
#. You are given a graph with nn nodes and mm edges, where each edge has a weight. Determine if there exists a subset of edges such that the graph becomes a tree and the sum of weights is odd.
#. You are tasked to partition a graph into two subgraphs such that the difference in the number of nodes between the two subgraphs is minimized.
#. In a large social network graph, find the smallest group of people (nodes) such that every other person in the network is directly connected to at least one person in this group.
#. Find the longest path in a Directed Acyclic Graph (DAG) where all nodes must be visited exactly once.
#. Given a weighted undirected graph, find the number of distinct Minimum Spanning Trees (MSTs) that can be formed.
#. You are given a graph where each node has a value. Find the largest sum of values that can be obtained by traversing from a given start node to an end node while following the graph’s edges.
#. You are given a directed graph representing a city's one-way road system. Each node represents an intersection, and each edge represents a road. Due to construction, one road (edge) can be closed. Determine whether the city remains fully connected (i.e., you can still reach all intersections from any starting intersection) if any one road is removed.
#. You are given an undirected graph representing a set of servers connected by cables. A server is considered critical if removing it causes some servers to become disconnected. Find all the critical servers in the graph.
#. A company wants to install a messaging system in its office building. The building is represented as a weighted undirected graph, where nodes are rooms and edges are connections between rooms. Messages can only travel over edges. Determine the minimum set of edges to remove such that there is no path between two specific rooms while keeping the rest of the graph connected.
#. You are given a directed acyclic graph (DAG) where each node represents a task, and each edge (u, v) means task u must be completed before task v. Multiple workers are available to work on tasks simultaneously. Each task takes exactly 1 unit of time to complete. Calculate the minimum time required to complete all tasks.
#. Given a grid with n rows and m columns, each cell is either land (1) or water (0). You can traverse only horizontally or vertically. A bridge can be built between two pieces of land separated by water if the Manhattan distance between them is 1. Determine the minimum number of bridges needed to connect all pieces of land into a single connected component.
#. A tournament is represented as a directed graph, where each edge (u, v) means team u defeated team v. Some match results are missing, represented as missing edges. Determine if it is possible to orient the missing edges such that the resulting graph is still a tournament.
#. You are given an undirected graph representing a city's sewer system, where nodes are sewer junctions and edges are pipes connecting them. Certain pipes are old and at risk of breaking. Find the minimum number of new pipes that need to be added to ensure that no single pipe failure disconnects any part of the system.
#. You are given a weighted undirected graph representing a network of computers. Some edges are "critical" (important for connectivity), and some are "pseudo-critical" (important but can be replaced by other edges). Write an algorithm to classify each edge as critical, pseudo-critical, or neither.
#. You are given a directed graph where each edge has an initial cost. You can choose to reduce the weight of up to :math:`k` edges by half. Find the minimum total cost to travel between two given nodes after applying this optimization.
#. You are given a directed graph where some edges have been removed, resulting in a disconnected graph. Determine the minimum number of edges to add back to restore strong connectivity.
#. You are given an undirected graph with :math:`n` nodes. The graph is subject to operations of two types: 1. Add an edge between two nodes. 2. Check if two nodes are in the same connected component. Implement an algorithm to handle these operations efficiently.
#. Given a directed acyclic graph (DAG) where each edge has a weight and a constraint :math:`k`, find the maximum sum of weights for any path containing at most :math:`k` edges.
#. A city is represented as a weighted grid where each cell has an elevation. Water floods from a source cell and can only flow to adjacent cells with equal or lower elevation. Determine the total area of cells that will be flooded.
#. You are given an undirected graph representing a network of roads between cities. A road is considered "critical" if removing it increases the shortest path between any two cities. Identify all critical roads in the graph.
#. You are given a directed graph with :math:`n` nodes and :math:`m` edges. Some edges are "mandatory," and others are "optional." Determine if it's possible to orient the optional edges to form a directed acyclic graph (DAG).
#. A company plans to expand its network by adding new connections. Each connection has a cost, and the company has a fixed budget. Find the maximum number of nodes that can be connected to the network within the budget.
#. You are given a directed graph where each node can serve as a starting point for spreading information. Calculate the minimum time required for information to reach all nodes, assuming it spreads simultaneously from all sources.
#. Given an undirected graph, color its nodes using the minimum number of colors such that no two adjacent nodes have the same color. Additionally, certain nodes have preassigned colors, and the coloring must respect these assignments.
#. You are given a directed graph where some nodes act as sources and others as sinks. Find the maximum flow in the network, assuming flow can originate from multiple sources and terminate at multiple sinks.
#. You are given a weighted undirected graph and a threshold :math:`t`. Form clusters by removing edges with weights greater than :math:`t`. Calculate the number of resulting clusters and the size of the largest cluster.
#. You are given a list of shortest paths between all pairs of nodes in an undirected graph. Determine if it is possible to reconstruct the original graph. If multiple graphs are possible, return any valid one.
#. You are given a directed graph where each edge has a delay time. Calculate the minimum total delay required to synchronize all nodes such that every node receives a signal at the same time.
#. A travel route is represented as a directed graph with costs on edges. You must visit certain mandatory nodes exactly once in any order. Find the shortest path that satisfies these constraints.
#. Given a directed graph, a source node, and a destination node, find the :math:`k`-th shortest path from the source to the destination.
#. You are given an undirected graph. Determine the minimum number of nodes that must be removed so that the remaining graph is still fully connected.
#. A road network is represented as a weighted undirected graph. Each road has a traffic limit. Determine if it is possible to reroute all vehicles such that the traffic on no road exceeds its limit.
#. You are given a weighted directed graph. Find the minimum weight cycle (if it exists) and return its weight. If no cycle exists, return -1.
#. You are given an undirected graph. Remove the minimum number of edges to partition the graph into two disjoint connected components of equal size (or as close as possible).

12. Multi-Step Problems
--------------------------------------------------------------------------------
#. Verifying and Improving Connectivity

	The police department in the city has converted every street into a one-way road. The mayor claims that it is possible to legally drive from any intersection in the city to any other intersection.
	
		* Verify Strong Connectivity: Design an algorithm to determine whether the city is strongly connected. If it is not, refute the mayor’s claim.  
		* Good Intersections: Call an intersection :math:`x` *good* if, for any intersection :math:`y` that one can legally reach from :math:`x`, it is possible to legally drive from :math:`y` back to :math:`x`. The mayor further claims that over 95% of the intersections in Sham-Poobanana are good. Devise an algorithm to verify or refute this claim.  
		* Reachability Pairs: Count the number of pairs of intersections :math:`(A, B)` where :math:`A` can reach :math:`B`, but :math:`B` cannot reach :math:`A`.  
		* Maximum Reachability Intersection: Find the intersection with the highest reachability, defined as the number of intersections reachable from it.  
		* Restoring Strong Connectivity: Determine the minimum number of streets that need to be converted back to two-way roads to make the city strongly connected.  
		* Signage Changes with Minimum Hires: People can be hired at intersections to convert roads back to two-way streets. They must obey traffic laws while doing so (i.e., they can only travel back on a street after making it two-way). Devise an efficient algorithm to minimize the number of people hired and provide an order of operations for each person to change signage.
