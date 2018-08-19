# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import os
import pickle
import math

class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []


    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        if not len(self.queue) == 0:
            return heapq.heappop(self.queue)
        else:
            return None
        raise NotImplementedError

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        for x,(y,z) in enumerate(self.queue):
            if z[1][-1] == node_id:
                del self.queue[x]
                return self.queue
        raise NotImplementedError

    def node_access(self, node_id, flag = 0):
        for i, x in enumerate(self.queue):
            if x[1][0] == node_id:
                if flag == 0:
                    return x
                else:
                    del self.queue[i]


    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        return heapq.heappush(self.queue, node)
#        raise NotImplementedError

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier = PriorityQueue()
    frontier.append((0,[start]))
    explored = []
    while True:
        if frontier.size() == 0:
          return []
        path_cost, path = frontier.pop()
        path_end_node = path[-1]
        explored.append(path_end_node)
        adj_nodes = graph[path_end_node] # .neighbors(path_end_node)
        for node in adj_nodes:
            expanded_path = list(path)
            if node == goal:  # goal test when node generated
                print(type(path))
                return path + [node] #but this is not best path
            if node not in explored:
                expanded_path.append(node)
                if expanded_path not in frontier:
                    exp_path_cost = expanded_path.__len__()
                    frontier.append((exp_path_cost, expanded_path))
                    print(frontier.queue)
    raise NotImplementedError

#
def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier = PriorityQueue()
    frontier_end = []
    frontier.append((0, [start]))
    explored = []

    while True:
        if frontier.size() == 0:
          return []
        path_cost, path = frontier.pop()
        path_end_node = path[-1]
        if path_end_node == goal:
            return path
        explored.append(path_end_node)
        adj_nodes = graph[path_end_node]
        for node in adj_nodes:
            expanded_path = list(path)
            expanded_path.append(node)
            exp_path_cost = path_cost + graph[path_end_node][node]['weight']
            if node not in explored and node not in frontier_end:
                frontier.append((exp_path_cost, expanded_path))
                frontier_end.append([node])
            elif node in frontier_end:
                for temp_cost, temp_path in frontier:
                    temp_start = temp_path[0]
                    temp_end = temp_path[-1]
                    if temp_start == expanded_path[0] and temp_end == expanded_path[-1]:
                        if exp_path_cost < temp_cost:
                            frontier.remove(node)
                            frontier.append((exp_path_cost, expanded_path))
                            break
    raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    def node_access(frontier_bwd, fwd_end_node):
        for i,x in enumerate(frontier_bwd.queue):
            if x[1][0] == fwd_end_node:
                    return x

    def remove(frontier_fwd, node_id):
        for x,(y,z) in enumerate(frontier_fwd.queue):
            if z[1][-1] == node_id:
                del frontier_fwd.queue[x]
                return frontier_fwd

    if start == goal:
        return []
    frontier_fwd = PriorityQueue()
    frontier_bwd = PriorityQueue()
    frontier_fwd.append((0, (start, [start])))
    frontier_bwd.append((0, (goal, [goal])))
    explored_fwd = []
    explored_bwd = []
    optim_path = []
    optim_cost = float("inf")

    while frontier_fwd.size() > 0 and frontier_bwd.size() > 0:
        fwd_path_cost, path_node = frontier_fwd.queue[0]
        fwd_end_node = path_node[0]
        fwd_path = path_node[1]
        bwd_path_cost, path_node = frontier_bwd.queue[0]

        if fwd_path_cost + bwd_path_cost >= optim_cost:
            print("cost-sum")
            print(fwd_path_cost + bwd_path_cost)
            print(optim_cost)
            return optim_path
        if fwd_path_cost <= bwd_path_cost:
            fwd_end_node = fwd_path[-1]
            if fwd_end_node == goal:
                if optim_cost > fwd_path_cost:
                    optim_cost = fwd_path_cost
                    optim_path = fwd_path
            elif fwd_end_node in [x[0] for _, x in frontier_bwd.queue]:
                bwd_path_cost, bwd_node = node_access(frontier_bwd, fwd_end_node)
                rev_path = list(reversed(bwd_node[1]))
                new_cost = fwd_path_cost + bwd_path_cost
                if optim_cost > new_cost:
                    optim_cost = new_cost
                    bwd_path = bwd_node[1]
                    optim_path = fwd_path + rev_path[1:len(bwd_path)]
                    print("rrrrrrr")
                    print(optim_path)
            # generic UCS part
            explored_fwd.append(fwd_end_node)
            frontier_fwd.pop()
            adj_nodes = graph[fwd_end_node]
            for node in adj_nodes:
                expanded_path_fwd = list(fwd_path)
                expanded_path_fwd.append(node)
                expanded_cost_fwd = fwd_path_cost + graph[fwd_end_node][node]['weight']
                if node not in explored_fwd and node not in [x[0] for _, x in frontier_fwd.queue]:
                    frontier_fwd.append((expanded_cost_fwd, (node, expanded_path_fwd)))
                elif node in [x[0] for _, x in frontier_fwd.queue]:
                    for temp_cost_fwd,temp_path_fwd in frontier_fwd:
                        temp_start_fwd = temp_path_fwd[1][0]
                        temp_end_fwd = temp_path_fwd[1][-1]
                        if temp_start_fwd == expanded_path_fwd[0] and temp_end_fwd == expanded_path_fwd[-1]:
                            if expanded_cost_fwd < temp_cost_fwd:
                                remove(frontier_fwd, node)
                                frontier_fwd.append((expanded_cost_fwd, (node, expanded_path_fwd)))
                                #print(optim_path)
                                break

        else: #Backward search starts here
            print("Backward")
            if frontier_bwd.size() > 0:
              bwd_path_cost, path_node = frontier_bwd.pop()
              bwd_end_node = path_node[0]
              bwd_path = path_node[1]
              print("bwd_path_cost")
              print(bwd_path_cost)
            if bwd_end_node == start:
              if optim_cost > bwd_path_cost:
                optim_cost = bwd_path_cost
                optim_path = list(reversed(bwd_path))
            elif bwd_end_node in [y[0] for _,y in frontier_fwd.queue]:
              fwd_path_cost, fwd_node = node_access(frontier_fwd, bwd_end_node) # do we remove this path from fwd
              new_cost = fwd_path_cost + bwd_path_cost
              rev_path = list(reversed(bwd_path))
              if optim_cost > new_cost:
                optim_cost = new_cost
                fwd_path = fwd_node[1]
                optim_path = fwd_path + rev_path[1:len(bwd_path)]
            # generic UCS part
            explored_bwd.append(bwd_end_node)
            adj_nodes = graph[bwd_end_node]
            for node in adj_nodes:
                expanded_path_bwd = list(bwd_path)
                expanded_path_bwd.append(node)
                expanded_cost_bwd = bwd_path_cost + graph[bwd_end_node][node]['weight']
                if node not in explored_bwd and node not in [x[0] for _, x in frontier_bwd.queue]:
                    frontier_bwd.append((expanded_cost_bwd, (node, expanded_path_bwd)))
                elif node in [x[0] for _, x in frontier_bwd.queue]:
                    for temp_cost_bwd, temp_path_bwd in frontier_bwd:
                        temp_start_bwd = temp_path_bwd[1][0]
                        temp_end_bwd = temp_path_bwd[1][-1]
                        if temp_start_bwd == expanded_path_bwd[0] and temp_end_bwd == expanded_path_bwd[-1]:
                            if expanded_cost_bwd < temp_cost_bwd:
                                remove(frontier_bwd, node)
                                frontier_bwd.append((expanded_cost_bwd, (node, expanded_path_bwd)))
                                break

    return optim_path
    raise NotImplementedError

def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    node = graph.node[v]
    goal_node = graph.node[goal]
    node_pos = node['pos']
    goal_pos = goal_node['pos']
    x = math.pow(node_pos[0] - goal_pos[0], 2)
    y = math.pow(node_pos[1] - goal_pos[1], 2)
    euclid_dist = math.sqrt(x+y)

    return euclid_dist
    raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    def calc_pathcost(graph, path):
        cost = 0
        path_len = path.__len__()
        if path_len == 0 or path_len == 1:
          return 0
        for i in range(0, path_len-1):
          node1 = path[i]
          node2 = path[i+1]
          cost = cost + graph[node1][node2]['weight']
        return cost
    
    def remove(frontier_fwd, node_id):
        for x,(y,z) in enumerate(frontier_fwd.queue):
            if z[1][-1] == node_id:
                del frontier_fwd.queue[x]
                return frontier_fwd

    if start == goal:
      return []
    frontier = PriorityQueue()
    frontier_end = []
    frontier.append((0, [start]))
    explored = []

    while True:
        if frontier.size() == 0:
          return []
        path_cost, path = frontier.pop()
        path_end_node = path[-1]
        if path_end_node == goal:
            return path

        explored.append(path_end_node)
        adj_nodes = graph[path_end_node]
        for node in adj_nodes:
            expanded_path = list(path)
            expanded_path.append(node)
            path_cost = calc_pathcost(graph, expanded_path)
            exp_path_cost = path_cost + heuristic(graph, node, goal)
            if node not in explored and node not in frontier_end:
                frontier.append((exp_path_cost, expanded_path))
                print("----")
                print(frontier)
                frontier_end.append([node])
            elif node in frontier_end:
                for temp_cost, temp_path in frontier:
                    temp_start = temp_path[0]
                    temp_end = temp_path[-1]
                    if temp_start == expanded_path[0] and temp_end == expanded_path[-1]:
                        if exp_path_cost < temp_cost:
                            remove(frontie, node)
                            frontier.append((exp_path_cost, expanded_path))
                            break

    raise NotImplementedError





def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    def node_access(frontier_bwd, fwd_end_node):
        for i,x in enumerate(frontier_bwd.queue):
            if x[1][0] == fwd_end_node:
                    return x

    def calc_pathcost(graph, path):
        cost = 0
        path_len = path.__len__()
        if path_len == 0 or path_len == 1:
          return 0
        for i in range(0, path_len-1):
          node1 = path[i]
          node2 = path[i+1]
          cost = cost + graph[node1][node2]['weight']
        return cost
    
    def remove(frontier_fwd, node_id):
        for x,(y,z) in enumerate(frontier_fwd.queue):
            if z[1][-1] == node_id:
                del frontier_fwd.queue[x]
                return frontier_fwd

    if start == goal:
        return []
    frontier_fwd = PriorityQueue()
    frontier_bwd = PriorityQueue()
    frontier_fwd.append((0, (start, [start])))
    frontier_bwd.append((0, (goal, [goal])))
    explored_fwd = []
    explored_bwd = []
    optim_path = []
    optim_cost = float("inf")

    while frontier_fwd.size() > 0 and frontier_bwd.size() > 0:
        fwd_path_cost, path_node = frontier_fwd.queue[0]
        fwd_end_node = path_node[0]
        fwd_path = path_node[1]
        bwd_path_cost, path_node = frontier_bwd.queue[0]

        if fwd_path_cost + bwd_path_cost >= optim_cost:
            return optim_path
        if fwd_path_cost <= bwd_path_cost:
            print("fwd_path_cost")
            print(fwd_path_cost)
            print("Forward")
            fwd_end_node = fwd_path[-1]
            if fwd_end_node == goal:
                if optim_cost > fwd_path_cost:
                    optim_cost = fwd_path_cost
                    optim_path = fwd_path
            elif fwd_end_node in [x[0] for _, x in frontier_bwd.queue]:
                bwd_path_cost, bwd_node = node_access(frontier_bwd, fwd_end_node)
                rev_path = list(reversed(bwd_node[1]))
                #new_cost = fwd_path_cost + bwd_path_cost
                comb_path = fwd_path + rev_path[1:len(bwd_node[1])]
                new_cost = calc_pathcost(graph, comb_path)
                if optim_cost > new_cost:
                    optim_cost = new_cost
                    bwd_path = bwd_node[1]
                    optim_path = fwd_path + rev_path[1:len(bwd_path)]
            # generic UCS part
            explored_fwd.append(fwd_end_node)
            frontier_fwd.pop()
            adj_nodes = graph[fwd_end_node]
            for node in adj_nodes:
                expanded_path_fwd = list(fwd_path)
                expanded_path_fwd.append(node)
                path_cost = calc_pathcost(graph, expanded_path_fwd)
                expanded_cost_fwd = path_cost + heuristic(graph, node, goal)
                if node not in explored_fwd and node not in [x[0] for _, x in frontier_fwd.queue]:
                    frontier_fwd.append((expanded_cost_fwd, (node, expanded_path_fwd)))
                elif node in [x[0] for _, x in frontier_fwd.queue]:
                    for temp_cost_fwd,temp_path_fwd in frontier_fwd:
                        temp_start_fwd = temp_path_fwd[1][0]
                        temp_end_fwd = temp_path_fwd[1][-1]
                        if temp_start_fwd == expanded_path_fwd[0] and temp_end_fwd == expanded_path_fwd[-1]:
                            if expanded_cost_fwd < temp_cost_fwd:
                                remove(frontier_fwd, node)
                                frontier_fwd.append((expanded_cost_fwd, (node, expanded_path_fwd)))
                                #print(optim_path)
                                break

        else: #Backward search starts here
            print("Backward")
            if frontier_bwd.size() > 0:
              bwd_path_cost, path_node = frontier_bwd.pop()
              bwd_end_node = path_node[0]
              bwd_path = path_node[1]
              print("bwd_path_cost")
              print(bwd_path_cost)
            if bwd_end_node == start:
              if optim_cost > bwd_path_cost:
                optim_cost = bwd_path_cost
                optim_path = list(reversed(bwd_path))
            elif bwd_end_node in [y[0] for _,y in frontier_fwd.queue]:
              fwd_path_cost, fwd_node = node_access(frontier_fwd, bwd_end_node) # do we remove this path from fwd
              #new_cost = fwd_path_cost + bwd_path_cost
              rev_path = list(reversed(bwd_path))
              comb_path = fwd_node[1] + rev_path[1:len(bwd_path)]
              new_cost = calc_pathcost(graph, comb_path)
              
              if optim_cost > new_cost:
                optim_cost = new_cost
                fwd_path = fwd_node[1]
                optim_path = fwd_path + rev_path[1:len(bwd_path)]
            # generic UCS part
            explored_bwd.append(bwd_end_node)
            adj_nodes = graph[bwd_end_node]
            for node in adj_nodes:
                expanded_path_bwd = list(bwd_path)
                expanded_path_bwd.append(node)
                path_cost = calc_pathcost(graph, expanded_path_bwd)
                expanded_cost_bwd = path_cost + heuristic(graph, node, start)
                if node not in explored_bwd and node not in [x[0] for _, x in frontier_bwd.queue]:
                    frontier_bwd.append((expanded_cost_bwd, (node, expanded_path_bwd)))
                elif node in [x[0] for _, x in frontier_bwd.queue]:
                    for temp_cost_bwd, temp_path_bwd in frontier_bwd:
                        temp_start_bwd = temp_path_bwd[1][0]
                        temp_end_bwd = temp_path_bwd[1][-1]
                        if temp_start_bwd == expanded_path_bwd[0] and temp_end_bwd == expanded_path_bwd[-1]:
                            if expanded_cost_bwd < temp_cost_bwd:
                                remove(frontier_bwd, node)
                                frontier_bwd.append((expanded_cost_bwd, (node, expanded_path_bwd)))
                                break

    return optim_path
    raise NotImplementedError


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    if goals[0] == goals[1] and goals[1] == goals[2]:
      return []

    print("GOALS")
    print(goals)
    def node_access(frontier_bwd, fwd_end_node):
        for i,x in enumerate(frontier_bwd.queue):
            if x[1][0] == fwd_end_node:
                    return x

    def remove(frontier_fwd, node_id):
        for x,(y,z) in enumerate(frontier_fwd.queue):
            if z[1][-1] == node_id:
                del frontier_fwd.queue[x]
                return frontier_fwd
            
    start = goals[0]
    mid = goals[1]
    goal = goals[2]
    l,m,n = [0,0,0]
    while True:
        if l == 0:
            if start == mid:
                l = 1
                optim_path_12 = [] 
                break
            
            frontier_s = PriorityQueue()
            frontier_m = PriorityQueue()
            frontier_s.append((0, (start, [start])))
            frontier_m.append((0, (mid, [mid])))
            explored_s = []
            explored_m = []
            optim_path_12 = []
            optim_cost_12 = float("inf")
            print("xxxxxxxxxxxxxx")
            
            if start == mid:
                l = 1
                optim_path_12 = [] 
                break
            
            while frontier_s.size() > 0 and frontier_m.size() > 0:
                s_path_cost, path_node = frontier_s.queue[0]
                s_end_node = path_node[0]
                s_path = path_node[1]
                m_path_cost, path_node = frontier_m.queue[0]
            
                if s_path_cost + m_path_cost >= optim_cost_12:
                    l = 1
                    break
                if s_path_cost <= m_path_cost:
                    print("s_path_cost")
                    print(s_path_cost)
                    print("start - s")
                    s_end_node = s_path[-1]
                    if s_end_node == mid:
                        if optim_cost_12 > s_path_cost:
                            optim_cost_12 = s_path_cost
                            optim_path_12 = s_path
                    elif s_end_node in [x[0] for _, x in frontier_m.queue]:
                        m_path_cost, m_node = node_access(frontier_m, s_end_node)
                        rev_path = list(reversed(m_node[1]))
                        new_cost = s_path_cost + m_path_cost
                        if optim_cost_12 > new_cost:
                            optim_cost_12 = new_cost
                            m_path = m_node[1]
                            optim_path_12 = s_path + rev_path[1:len(m_path)]
                    # generic UCS part
                    explored_s.append(s_end_node)
                    frontier_s.pop()
                    adj_nodes = graph[s_end_node]
                    for node in adj_nodes:
                        expanded_path_s = list(s_path)
                        expanded_path_s.append(node)
                        expanded_cost_s = s_path_cost + graph[s_end_node][node]['weight']
                        if node not in explored_s and node not in [x[0] for _, x in frontier_s.queue]:
                            frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                        elif node in [x[0] for _, x in frontier_s.queue]:
                            for temp_cost_s,temp_path_s in frontier_s:
                                temp_start_s = temp_path_s[1][0]
                                temp_end_s = temp_path_s[1][-1]
                                if temp_start_s == expanded_path_s[0] and temp_end_s == expanded_path_s[-1]:
                                    if expanded_cost_s < temp_cost_s:
                                        remove(frontier_s, node)
                                        frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                                        #print(optim_path)
                                        break
            
                else: #Backward search starts here
                    print("Backward - m")
                    if frontier_m.size() > 0:
                      m_path_cost, path_node = frontier_m.pop()
                      m_end_node = path_node[0]
                      m_path = path_node[1]
                      print("m_path_cost")
                      print(m_path_cost)
                    if m_end_node == start:
                      if optim_cost_12 > m_path_cost:
                        optim_cost_12 = m_path_cost
                        optim_path_12 = list(reversed(m_path))
                    elif m_end_node in [y[0] for _,y in frontier_s.queue]:
                      s_path_cost, s_node = node_access(frontier_s, m_end_node) # do we remove this path from fwd
                      new_cost = s_path_cost + m_path_cost
                      rev_path = list(reversed(m_path))
                      if optim_cost_12 > new_cost:
                        optim_cost_12 = new_cost
                        s_path = s_node[1]
                        optim_path_12 = s_path + rev_path[1:len(m_path)]
                    # generic UCS part
                    explored_m.append(m_end_node)
                    adj_nodes = graph[m_end_node]
                    for node in adj_nodes:
                        expanded_path_m = list(m_path)
                        expanded_path_m.append(node)
                        expanded_cost_m = m_path_cost + graph[m_end_node][node]['weight']
                        if node not in explored_m and node not in [x[0] for _, x in frontier_m.queue]:
                            frontier_m.append((expanded_cost_m, (node, expanded_path_m)))
                        elif node in [x[0] for _, x in frontier_m.queue]:
                            for temp_cost_m, temp_path_m in frontier_m:
                                temp_start_m = temp_path_m[1][0]
                                temp_end_m = temp_path_m[1][-1]
                                if temp_start_m == expanded_path_m[0] and temp_end_m == expanded_path_m[-1]:
                                    if expanded_cost_m < temp_cost_m:
                                        remove(frontier_m, node)
                                        frontier_m.append((expanded_cost_m, (node, expanded_path_m)))
                                        break
        print("result start - mid")
        print(optim_path_12)
        print(optim_cost_12)
        if m == 0:
            print("y -started")
            frontier_s = PriorityQueue()
            frontier_g = PriorityQueue()
            frontier_s.append((0, (start, [start])))
            frontier_g.append((0, (goal, [goal])))
            explored_s = []
            explored_g = []
            optim_path_13 = []
            optim_cost_13 = float("inf")
            print("yyyyyyyyyyyyyyyyyyyyy")
            if start == goal:
                m = 1
                optim_path_13 = [] 
                break
            
            while frontier_s.size() > 0 and frontier_g.size() > 0:
                s_path_cost, path_node = frontier_s.queue[0]
                s_end_node = path_node[0]
                s_path = path_node[1]
                g_path_cost, path_node = frontier_g.queue[0]
            
                if s_path_cost + g_path_cost >= optim_cost_13:
                    m = 1
                    break
                if s_path_cost <= g_path_cost:
                    print("s_path_cost")
                    print(g_path_cost)
                    print("start - s")
                    s_end_node = s_path[-1]
                    if s_end_node == goal:
                        if optim_cost_13 > s_path_cost:
                            optim_cost_13 = s_path_cost
                            optim_path_13 = s_path
                    elif s_end_node in [x[0] for _, x in frontier_g.queue]:
                        g_path_cost, g_node = node_access(frontier_g, s_end_node)
                        rev_path = list(reversed(g_node[1]))
                        new_cost = s_path_cost + g_path_cost
                        if optim_cost_13 > new_cost:
                            optim_cost_13 = new_cost
                            g_path = g_node[1]
                            optim_path_13 = s_path + rev_path[1:len(g_path)]
                    # generic UCS part
                    explored_s.append(s_end_node)
                    frontier_s.pop()
                    adj_nodes = graph[s_end_node]
                    for node in adj_nodes:
                        expanded_path_s = list(s_path)
                        expanded_path_s.append(node)
                        expanded_cost_s = s_path_cost + graph[s_end_node][node]['weight']
                        if node not in explored_s and node not in [x[0] for _, x in frontier_s.queue]:
                            frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                        elif node in [x[0] for _, x in frontier_s.queue]:
                            for temp_cost_s,temp_path_s in frontier_s:
                                temp_start_s = temp_path_s[1][0]
                                temp_end_s = temp_path_s[1][-1]
                                if temp_start_s == expanded_path_s[0] and temp_end_s == expanded_path_s[-1]:
                                    if expanded_cost_s < temp_cost_s:
                                        remove(frontier_s, node)
                                        frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                                        #print(optim_path)
                                        break
            
                else: #Backward search starts here
                    print("Backward - g")
                    if frontier_g.size() > 0:
                      g_path_cost, path_node = frontier_g.pop()
                      g_end_node = path_node[0]
                      g_path = path_node[1]
                      print("m_path_cost")
                      print(g_path_cost)
                    if g_end_node == start:
                      if optim_cost_13 > g_path_cost:
                        optim_cost_13 = g_path_cost
                        optim_path_13 = list(reversed(g_path))
                    elif g_end_node in [y[0] for _,y in frontier_s.queue]:
                      s_path_cost, s_node = node_access(frontier_s, g_end_node) # do we remove this path from fwd
                      new_cost = s_path_cost + g_path_cost
                      rev_path = list(reversed(g_path))
                      if optim_cost_13 > new_cost:
                        optim_cost_13 = new_cost
                        s_path = s_node[1]
                        optim_path_13 = s_path + rev_path[1:len(g_path)]
                    # generic UCS part
                    explored_g.append(g_end_node)
                    adj_nodes = graph[g_end_node]
                    for node in adj_nodes:
                        expanded_path_g = list(g_path)
                        expanded_path_g.append(node)
                        expanded_cost_g = g_path_cost + graph[g_end_node][node]['weight']
                        if node not in explored_g and node not in [x[0] for _, x in frontier_g.queue]:
                            frontier_g.append((expanded_cost_g, (node, expanded_path_g)))
                        elif node in [x[0] for _, x in frontier_g.queue]:
                            for temp_cost_g, temp_path_g in frontier_g:
                                temp_start_g = temp_path_g[1][0]
                                temp_end_g = temp_path_g[1][-1]
                                if temp_start_g == expanded_path_g[0] and temp_end_g == expanded_path_g[-1]:
                                    if expanded_cost_g < temp_cost_g:
                                        remove(frontier_g, node)
                                        frontier_g.append((expanded_cost_g, (node, expanded_path_g)))
                                        break
        print("result start - goal")
        print(optim_path_13)
        print(optim_cost_13)
        if n == 0:
            frontier_s = PriorityQueue()
            frontier_g = PriorityQueue()
            frontier_s.append((0, (mid, [mid])))
            frontier_g.append((0, (goal, [goal])))
            explored_s = []
            explored_g = []
            optim_path_23 = []
            optim_cost_23 = float("inf")
            print("zzzzzzzzzz")
            
            if start == goal:
                n = 1
                optim_path_23 = [] 
                break
            
            while frontier_s.size() > 0 and frontier_g.size() > 0:
                s_path_cost, path_node = frontier_s.queue[0]
                s_end_node = path_node[0]
                s_path = path_node[1]
                g_path_cost, path_node = frontier_g.queue[0]
            
                if s_path_cost + g_path_cost >= optim_cost_23:
                    n = 1
                    break
                if s_path_cost <= g_path_cost:
                    print("s_path_cost")
                    print(g_path_cost)
                    print("start - s")
                    s_end_node = s_path[-1]
                    if s_end_node == goal:
                        if optim_cost_23 > s_path_cost:
                            optim_cost_23 = s_path_cost
                            optim_path_23 = s_path
                    elif s_end_node in [x[0] for _, x in frontier_g.queue]:
                        g_path_cost, g_node = node_access(frontier_g, s_end_node)
                        rev_path = list(reversed(g_node[1]))
                        new_cost = s_path_cost + g_path_cost
                        if optim_cost_23 > new_cost:
                            optim_cost_23 = new_cost
                            g_path = g_node[1]
                            optim_path_23 = s_path + rev_path[1:len(g_path)]
                    # generic UCS part
                    explored_s.append(s_end_node)
                    frontier_s.pop()
                    adj_nodes = graph[s_end_node]
                    for node in adj_nodes:
                        expanded_path_s = list(s_path)
                        expanded_path_s.append(node)
                        expanded_cost_s = s_path_cost + graph[s_end_node][node]['weight']
                        if node not in explored_s and node not in [x[0] for _, x in frontier_s.queue]:
                            frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                        elif node in [x[0] for _, x in frontier_s.queue]:
                            for temp_cost_s,temp_path_s in frontier_s:
                                temp_start_s = temp_path_s[1][0]
                                temp_end_s = temp_path_s[1][-1]
                                if temp_start_s == expanded_path_s[0] and temp_end_s == expanded_path_s[-1]:
                                    if expanded_cost_s < temp_cost_s:
                                        remove(frontier_s, node)
                                        frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                                        #print(optim_path)
                                        break
            
                else: #Backward search starts here
                    print("Backward - g")
                    if frontier_g.size() > 0:
                      g_path_cost, path_node = frontier_g.pop()
                      g_end_node = path_node[0]
                      g_path = path_node[1]
                      print("m_path_cost")
                      print(g_path_cost)
                    if g_end_node == mid:
                      if optim_cost_23 > g_path_cost:
                        optim_cost_23 = g_path_cost
                        optim_path_23 = list(reversed(g_path))
                    elif g_end_node in [y[0] for _,y in frontier_s.queue]:
                      s_path_cost, s_node = node_access(frontier_s, g_end_node) # do we remove this path from fwd
                      new_cost = s_path_cost + g_path_cost
                      rev_path = list(reversed(g_path))
                      if optim_cost_23 > new_cost:
                        optim_cost_23 = new_cost
                        s_path = s_node[1]
                        optim_path_23 = s_path + rev_path[1:len(g_path)]
                    # generic UCS part
                    explored_g.append(g_end_node)
                    adj_nodes = graph[g_end_node]
                    for node in adj_nodes:
                        expanded_path_g = list(g_path)
                        expanded_path_g.append(node)
                        expanded_cost_g = g_path_cost + graph[g_end_node][node]['weight']
                        if node not in explored_g and node not in [x[0] for _, x in frontier_g.queue]:
                            frontier_g.append((expanded_cost_g, (node, expanded_path_g)))
                        elif node in [x[0] for _, x in frontier_g.queue]:
                            for temp_cost_g, temp_path_g in frontier_g:
                                temp_start_g = temp_path_g[1][0]
                                temp_end_g = temp_path_g[1][-1]
                                if temp_start_g == expanded_path_g[0] and temp_end_g == expanded_path_g[-1]:
                                    if expanded_cost_g < temp_cost_g:
                                        remove(frontier_g, node)
                                        frontier_g.append((expanded_cost_g, (node, expanded_path_g)))
                                        break
        
        print("result mid - goal")
        print(optim_cost_23)
        cost_list = [optim_cost_12, optim_cost_23, optim_cost_13]
        print("12")
        print(optim_path_12)
        print(optim_cost_12)
        print("23")
        print(optim_path_23)
        print(optim_cost_23)
        print("13")
        print(optim_path_13)
        print(optim_cost_13)
        max_cost = cost_list.index(max(cost_list))
        if max_cost == 0:
            rev_path = list(reversed(optim_path_13))
            final_path = optim_path_23 +  rev_path[1:len(optim_path_13)]
        elif max_cost ==1:
            rev_path = list(reversed(optim_path_13))
            final_path = rev_path +  optim_path_12[1:len(optim_path_12)]
        else:
            final_path = optim_path_12 +  optim_path_23[1:len(optim_path_23)]
        print("final_path")
        print(final_path)
        return final_path

    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    if goals[0] == goals[1] and goals[1] == goals[2]:
      return []

    def node_access(frontier_bwd, fwd_end_node):
        for i,x in enumerate(frontier_bwd.queue):
            if x[1][0] == fwd_end_node:
                    return x

    def remove(frontier_fwd, node_id):
        for x,(y,z) in enumerate(frontier_fwd.queue):
            if z[1][-1] == node_id:
                del frontier_fwd.queue[x]
                return frontier_fwd
            
    start = goals[0]
    mid = goals[1]
    goal = goals[2]
    l,m,n = [0,0,0]
    while True:
        if l == 0:
            if start == mid:
                l = 1
                optim_path_12 = [] 
                break
            
            frontier_s = PriorityQueue()
            frontier_m = PriorityQueue()
            frontier_s.append((0, (start, [start])))
            frontier_m.append((0, (mid, [mid])))
            explored_s = []
            explored_m = []
            optim_path_12 = []
            optim_cost_12 = float("inf")
            if start == mid:
                l = 1
                optim_path_12 = [] 
                break
            
            while frontier_s.size() > 0 and frontier_m.size() > 0:
                s_path_cost, path_node = frontier_s.queue[0]
                s_end_node = path_node[0]
                s_path = path_node[1]
                m_path_cost, path_node = frontier_m.queue[0]
            
                if s_path_cost + m_path_cost >= optim_cost_12:
                    l = 1
                    break
                if s_path_cost <= m_path_cost:
                    s_end_node = s_path[-1]
                    if s_end_node == mid:
                        if optim_cost_12 > s_path_cost:
                            optim_cost_12 = s_path_cost
                            optim_path_12 = s_path
                    elif s_end_node in [x[0] for _, x in frontier_m.queue]:
                        m_path_cost, m_node = node_access(frontier_m, s_end_node)
                        rev_path = list(reversed(m_node[1]))
                        new_cost = s_path_cost + m_path_cost
                        if optim_cost_12 > new_cost:
                            optim_cost_12 = new_cost
                            m_path = m_node[1]
                            optim_path_12 = s_path + rev_path[1:len(m_path)]
                    # generic UCS part
                    explored_s.append(s_end_node)
                    frontier_s.pop()
                    adj_nodes = graph[s_end_node]
                    for node in adj_nodes:
                        expanded_path_s = list(s_path)
                        expanded_path_s.append(node)
                        expanded_cost_s = s_path_cost + graph[s_end_node][node]['weight']
                        if node not in explored_s and node not in [x[0] for _, x in frontier_s.queue]:
                            frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                        elif node in [x[0] for _, x in frontier_s.queue]:
                            for temp_cost_s,temp_path_s in frontier_s:
                                temp_start_s = temp_path_s[1][0]
                                temp_end_s = temp_path_s[1][-1]
                                if temp_start_s == expanded_path_s[0] and temp_end_s == expanded_path_s[-1]:
                                    if expanded_cost_s < temp_cost_s:
                                        remove(frontier_s, node)
                                        frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                                        #print(optim_path)
                                        break
            
                else: #Backward search starts here
                    if frontier_m.size() > 0:
                      m_path_cost, path_node = frontier_m.pop()
                      m_end_node = path_node[0]
                      m_path = path_node[1]
                    if m_end_node == start:
                      if optim_cost_12 > m_path_cost:
                        optim_cost_12 = m_path_cost
                        optim_path_12 = list(reversed(m_path))
                    elif m_end_node in [y[0] for _,y in frontier_s.queue]:
                      s_path_cost, s_node = node_access(frontier_s, m_end_node) # do we remove this path from fwd
                      new_cost = s_path_cost + m_path_cost
                      rev_path = list(reversed(m_path))
                      if optim_cost_12 > new_cost:
                        optim_cost_12 = new_cost
                        s_path = s_node[1]
                        optim_path_12 = s_path + rev_path[1:len(m_path)]
                    # generic UCS part
                    explored_m.append(m_end_node)
                    adj_nodes = graph[m_end_node]
                    for node in adj_nodes:
                        expanded_path_m = list(m_path)
                        expanded_path_m.append(node)
                        expanded_cost_m = m_path_cost + graph[m_end_node][node]['weight']
                        if node not in explored_m and node not in [x[0] for _, x in frontier_m.queue]:
                            frontier_m.append((expanded_cost_m, (node, expanded_path_m)))
                        elif node in [x[0] for _, x in frontier_m.queue]:
                            for temp_cost_m, temp_path_m in frontier_m:
                                temp_start_m = temp_path_m[1][0]
                                temp_end_m = temp_path_m[1][-1]
                                if temp_start_m == expanded_path_m[0] and temp_end_m == expanded_path_m[-1]:
                                    if expanded_cost_m < temp_cost_m:
                                        remove(frontier_m, node)
                                        frontier_m.append((expanded_cost_m, (node, expanded_path_m)))
                                        break

        if m == 0:
            frontier_s = PriorityQueue()
            frontier_g = PriorityQueue()
            frontier_s.append((0, (start, [start])))
            frontier_g.append((0, (goal, [goal])))
            explored_s = []
            explored_g = []
            optim_path_13 = []
            optim_cost_13 = float("inf")
            if start == goal:
                m = 1
                optim_path_13 = [] 
                break
            
            while frontier_s.size() > 0 and frontier_g.size() > 0:
                s_path_cost, path_node = frontier_s.queue[0]
                s_end_node = path_node[0]
                s_path = path_node[1]
                g_path_cost, path_node = frontier_g.queue[0]
            
                if s_path_cost + g_path_cost >= optim_cost_13:
                    m = 1
                    break
                if s_path_cost <= g_path_cost:
                    s_end_node = s_path[-1]
                    if s_end_node == goal:
                        if optim_cost_13 > s_path_cost:
                            optim_cost_13 = s_path_cost
                            optim_path_13 = s_path
                    elif s_end_node in [x[0] for _, x in frontier_g.queue]:
                        g_path_cost, g_node = node_access(frontier_g, s_end_node)
                        rev_path = list(reversed(g_node[1]))
                        new_cost = s_path_cost + g_path_cost
                        if optim_cost_13 > new_cost:
                            optim_cost_13 = new_cost
                            g_path = g_node[1]
                            optim_path_13 = s_path + rev_path[1:len(g_path)]
                    # generic UCS part
                    explored_s.append(s_end_node)
                    frontier_s.pop()
                    adj_nodes = graph[s_end_node]
                    for node in adj_nodes:
                        expanded_path_s = list(s_path)
                        expanded_path_s.append(node)
                        expanded_cost_s = s_path_cost + graph[s_end_node][node]['weight']
                        if node not in explored_s and node not in [x[0] for _, x in frontier_s.queue]:
                            frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                        elif node in [x[0] for _, x in frontier_s.queue]:
                            for temp_cost_s,temp_path_s in frontier_s:
                                temp_start_s = temp_path_s[1][0]
                                temp_end_s = temp_path_s[1][-1]
                                if temp_start_s == expanded_path_s[0] and temp_end_s == expanded_path_s[-1]:
                                    if expanded_cost_s < temp_cost_s:
                                        remove(frontier_s, node)
                                        frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                                        #print(optim_path)
                                        break
            
                else: #Backward search starts here
                    if frontier_g.size() > 0:
                      g_path_cost, path_node = frontier_g.pop()
                      g_end_node = path_node[0]
                      g_path = path_node[1]
                    if g_end_node == start:
                      if optim_cost_13 > g_path_cost:
                        optim_cost_13 = g_path_cost
                        optim_path_13 = list(reversed(g_path))
                    elif g_end_node in [y[0] for _,y in frontier_s.queue]:
                      s_path_cost, s_node = node_access(frontier_s, g_end_node) # do we remove this path from fwd
                      new_cost = s_path_cost + g_path_cost
                      rev_path = list(reversed(g_path))
                      if optim_cost_13 > new_cost:
                        optim_cost_13 = new_cost
                        s_path = s_node[1]
                        optim_path_13 = s_path + rev_path[1:len(g_path)]
                    # generic UCS part
                    explored_g.append(g_end_node)
                    adj_nodes = graph[g_end_node]
                    for node in adj_nodes:
                        expanded_path_g = list(g_path)
                        expanded_path_g.append(node)
                        expanded_cost_g = g_path_cost + graph[g_end_node][node]['weight']
                        if node not in explored_g and node not in [x[0] for _, x in frontier_g.queue]:
                            frontier_g.append((expanded_cost_g, (node, expanded_path_g)))
                        elif node in [x[0] for _, x in frontier_g.queue]:
                            for temp_cost_g, temp_path_g in frontier_g:
                                temp_start_g = temp_path_g[1][0]
                                temp_end_g = temp_path_g[1][-1]
                                if temp_start_g == expanded_path_g[0] and temp_end_g == expanded_path_g[-1]:
                                    if expanded_cost_g < temp_cost_g:
                                        remove(frontier_g, node)
                                        frontier_g.append((expanded_cost_g, (node, expanded_path_g)))
                                        break
        if n == 0:
            frontier_s = PriorityQueue()
            frontier_g = PriorityQueue()
            frontier_s.append((0, (mid, [mid])))
            frontier_g.append((0, (goal, [goal])))
            explored_s = []
            explored_g = []
            optim_path_23 = []
            optim_cost_23 = float("inf")

            if start == goal:
                n = 1
                optim_path_23 = [] 
                break
            
            while frontier_s.size() > 0 and frontier_g.size() > 0:
                s_path_cost, path_node = frontier_s.queue[0]
                s_end_node = path_node[0]
                s_path = path_node[1]
                g_path_cost, path_node = frontier_g.queue[0]
            
                if s_path_cost + g_path_cost >= optim_cost_23:
                    n = 1
                    break
                if s_path_cost <= g_path_cost:
                    s_end_node = s_path[-1]
                    if s_end_node == goal:
                        if optim_cost_23 > s_path_cost:
                            optim_cost_23 = s_path_cost
                            optim_path_23 = s_path
                    elif s_end_node in [x[0] for _, x in frontier_g.queue]:
                        g_path_cost, g_node = node_access(frontier_g, s_end_node)
                        rev_path = list(reversed(g_node[1]))
                        new_cost = s_path_cost + g_path_cost
                        if optim_cost_23 > new_cost:
                            optim_cost_23 = new_cost
                            g_path = g_node[1]
                            optim_path_23 = s_path + rev_path[1:len(g_path)]
                    # generic UCS part
                    explored_s.append(s_end_node)
                    frontier_s.pop()
                    adj_nodes = graph[s_end_node]
                    for node in adj_nodes:
                        expanded_path_s = list(s_path)
                        expanded_path_s.append(node)
                        expanded_cost_s = s_path_cost + graph[s_end_node][node]['weight']
                        if node not in explored_s and node not in [x[0] for _, x in frontier_s.queue]:
                            frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                        elif node in [x[0] for _, x in frontier_s.queue]:
                            for temp_cost_s,temp_path_s in frontier_s:
                                temp_start_s = temp_path_s[1][0]
                                temp_end_s = temp_path_s[1][-1]
                                if temp_start_s == expanded_path_s[0] and temp_end_s == expanded_path_s[-1]:
                                    if expanded_cost_s < temp_cost_s:
                                        remove(frontier_s, node)
                                        frontier_s.append((expanded_cost_s, (node, expanded_path_s)))
                                        #print(optim_path)
                                        break
            
                else: #Backward search starts here
                    if frontier_g.size() > 0:
                      g_path_cost, path_node = frontier_g.pop()
                      g_end_node = path_node[0]
                      g_path = path_node[1]
                    if g_end_node == mid:
                      if optim_cost_23 > g_path_cost:
                        optim_cost_23 = g_path_cost
                        optim_path_23 = list(reversed(g_path))
                    elif g_end_node in [y[0] for _,y in frontier_s.queue]:
                      s_path_cost, s_node = node_access(frontier_s, g_end_node) # do we remove this path from fwd
                      new_cost = s_path_cost + g_path_cost
                      rev_path = list(reversed(g_path))
                      if optim_cost_23 > new_cost:
                        optim_cost_23 = new_cost
                        s_path = s_node[1]
                        optim_path_23 = s_path + rev_path[1:len(g_path)]
                    # generic UCS part
                    explored_g.append(g_end_node)
                    adj_nodes = graph[g_end_node]
                    for node in adj_nodes:
                        expanded_path_g = list(g_path)
                        expanded_path_g.append(node)
                        expanded_cost_g = g_path_cost + graph[g_end_node][node]['weight']
                        if node not in explored_g and node not in [x[0] for _, x in frontier_g.queue]:
                            frontier_g.append((expanded_cost_g, (node, expanded_path_g)))
                        elif node in [x[0] for _, x in frontier_g.queue]:
                            for temp_cost_g, temp_path_g in frontier_g:
                                temp_start_g = temp_path_g[1][0]
                                temp_end_g = temp_path_g[1][-1]
                                if temp_start_g == expanded_path_g[0] and temp_end_g == expanded_path_g[-1]:
                                    if expanded_cost_g < temp_cost_g:
                                        remove(frontier_g, node)
                                        frontier_g.append((expanded_cost_g, (node, expanded_path_g)))
                                        break
        print(optim_cost_23)
        cost_list = [optim_cost_12, optim_cost_23, optim_cost_13]
        max_cost = cost_list.index(max(cost_list))
        
        if optim_path_12 == float('inf'):
            return optim_path_13
        elif optim_path_23 == float('inf'):
            return optim_path_12
        elif optim_path_13 == float('inf'):
            return optim_path_23
        if max_cost == 0:
            rev_path = list(reversed(optim_path_13))
            final_path = optim_path_23 +  rev_path[1:len(optim_path_13)]
        elif max_cost ==1:
            rev_path = list(reversed(optim_path_13))
            final_path = rev_path +  optim_path_12[1:len(optim_path_12)]
        else:
            final_path = optim_path_12 +  optim_path_23[1:len(optim_path_23)]
        return final_path

    raise NotImplementedError


def return_your_name():
    name = "Rajesh Pothamsetty"
    return name
    raise NotImplementedError


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data
