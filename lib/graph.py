import networkx as nx
import numpy as np
import itertools
from copy import deepcopy
from typing import Tuple, List, Set
from random import shuffle


def containsCycle(graph: nx.Graph) -> bool:
    """Given a networkx directed/undirected graph, determine whether a cycle exists in the graph.

    Args:
        graph (nx.Graph): networkx graph object

    Returns:
        bool: True if a cycle exists, false if otherwise
    """
    try:
        nx.find_cycle(graph)
        return True
    except nx.exception.NetworkXNoCycle:
        return False


def containsClique(graph: nx.Graph, greater_than_size: int = 2) -> bool:
    """Check if a undirected graph contains any cliques larger than a 
    specified size and returns True if true.

    Args:
        graph (nx.Graph): undirected graph
        greater_than_size (int, optional): maximum permissible clique. Defaults to 2.

    Returns:
        bool: True if graph contains clique
    """
    for subclique in nx.find_cliques(graph):
        if len(subclique) > greater_than_size:
            return True
    return False


def simplifyDG(graph: nx.DiGraph) -> nx.DiGraph:
    """Given a directed graph, reduce long chains with no siblings and 
    spouses per node to reduce plotting complexity.

    Args:
        graph (nx.DiGraph): networkx digraph

    Returns:
        nx.DiGraph: networkx digraph with pruned edges where nothing has happened for a while
        
    """
    outputGraph = deepcopy(graph)
    removed_nodes = []
    for node, _ in outputGraph.nodes(data=True):
        predecessors = list(outputGraph.predecessors(node))
        successors = list(outputGraph.successors(node))
        if len(predecessors) == 0 or len(successors) ==0:
            continue
        siblings = list(outputGraph.successors(predecessors[0]))
        siblings.remove(node)
        spouses = list(outputGraph.predecessors(successors[0]))
        spouses.remove(node)
        
        # If a node has one predecessor and one successor
        if len(predecessors) == 1 and len(successors) == 1:
            if siblings and spouses:
                pass
            else:
                outputGraph.add_edge(predecessors[0], successors[0])
                outputGraph.remove_edge(predecessors[0], node)
                outputGraph.remove_edge(node, successors[0])
                removed_nodes.append(node)
        elif len(predecessors) > 1 and len(successors) == 1:
            if siblings and spouses:
                pass
            else:
                for predecessor_ID in predecessors:
                    outputGraph.add_edge(predecessor_ID, successors[0])
                    outputGraph.remove_edge(predecessor_ID, node)
                outputGraph.remove_edge(node, successors[0])
                removed_nodes.append(node)
        # elif len(successors) > 1 and len(predecessors) == 1:
        #     if siblings and spouses:
        #         pass
        #     else:
        #         for successor_ID in successors:
        #             outputGraph.add_edge(predecessors[0], successor_ID)
        #             outputGraph.remove_edge(successor_ID, node)
        #         outputGraph.remove_edge(node, successors[0])
        #         removed_nodes.append(node)
            
    for nodeID in removed_nodes:
        outputGraph.remove_node(nodeID)
    return outputGraph


def getLargestSubnet(graph: nx.Graph) -> nx.Graph:
    """Given a networkx undirected graph, get the largest subgraph
    contained within graph by node count.

    Args:
        graph (nx.Graph): input networkx undirected graph

    Returns:
        nx.Graph: largest subgraph of graph by node count
    """
    connectedComponents = [c for c in sorted(
        nx.connected_components(graph), key=len, reverse=True)]
    return graph.subgraph(connectedComponents[0])


def getLongestChain(graph: nx.Graph) -> Tuple[set, int]:
    """Given an undirected graph, get the longest chain starting
    with a leaf node in the graph.

    Args:
        graph (nx.Graph): undirected graph to query

    Returns:
        Tuple[set, int]: list of node IDs associated with the longest chain (no particular order), 
            followed by  the node ID of the terminating leaf.
    """
    temp_graph = deepcopy(graph)

    hub_nodes = [node for (node, degree) in temp_graph.degree() if degree > 2]
    edges_2_rm = [i for n in hub_nodes for i in temp_graph.edges(n)]

    temp_graph.remove_edges_from(edges_2_rm)
    connectedComponents = [c for c in sorted(
        nx.connected_components(temp_graph), key=len, reverse=True)]
    for chain in connectedComponents:
        for id in chain:
            if graph.degree[id] == 1:
                return (chain, id)
    return None


def getLargestComponentSize(graph: nx.Graph) -> int:
    """Get the largest isolated subgraph size in the queried graph

    Args:
        graph (nx.Graph): queried graph

    Returns:
        int: number of nodes in the largest subgraph
    """
    connectedComponents = [c for c in sorted(
        nx.connected_components(graph), key=len, reverse=True)]
    return len(connectedComponents[0])


def getChainLength(graph: nx.Graph) -> List[int]:
    """Given a graph to query, identify the length of all chains
    in the graph. A chain is defined as a series of connected nodes
    that do not have degrees greater than 2.

    Args:
        graph (nx.Graph): graph to query

    Returns:
        List[int]: list containing length of each chain found in the
            queried graph. 
    """
    temp_graph = deepcopy(graph)
    remove = [node for node, degree in dict(
        temp_graph.degree()).items() if degree > 2]
    temp_graph.remove_nodes_from(remove)
    connectedComponents = [c for c in sorted(
        nx.connected_components(temp_graph), key=len, reverse=True)]
    data = []
    for chain in connectedComponents:
        data += [len(chain)]
    return data


def getInheritedChain(graph: nx.Graph, length: int, stop_if_branch: bool = False) -> Set[int]:
    """Helper function for Simulation module in simulation_handler.py
    Used to determine what part of the network structure is considered to be inherited

    Looks for the longest existing chain in the graph, and traces back from the tip of 
    the chain <length> units back. Chain is then severed there, and the fragment's 
    particleIDs are returned

    Args:
        graph (nx.Graph): simulation network structure in the form of an undirected graph
        length (int): length of chain to pass down
        stop_if_branch (bool): set to True so that if while traversing up the chain
            from the tail, if a branch is encountered, 

    Returns:
        Set[int]: particle IDs part of the chain.
    """
    idx2pID = {i: pID for i, pID in enumerate(sorted(graph.nodes))}
    pID2idx = {k: v for v, k in idx2pID.items()}
    # Get the longest existing chain in a structure.
    _, tipID = getLongestChain(graph)
    # Generate traceback graph from selected tipID
    traceback_graph = nx.DiGraph()
    for src, dst in nx.dfs_edges(graph, tipID):
        traceback_graph.add_edge(src, dst)
    adj_Mx = nx.to_numpy_array(
        traceback_graph, nodelist=sorted(traceback_graph.nodes))
    # Identify nodes that are 'length' nodes away from tipID node,
    # if none are found, reduce length by 1 unit until search is satisfied.
    while True:
        l_steps_away = np.linalg.matrix_power(adj_Mx, length)
        qualified_paths = l_steps_away.nonzero()
        re_qualified_paths = [(idx2pID[s], idx2pID[d]) for s, d in zip(
            qualified_paths[0], qualified_paths[-1]) if s == pID2idx[tipID]]

        if re_qualified_paths:
            break
        else:
            length -= 1

    rm_edges = []
    for src, dest in re_qualified_paths:
        for i, j in traceback_graph.edges(dest):
            rm_edges.append((i, j))
    traceback_graph.remove_edges_from(rm_edges)

    # Condition for if there are additional branches, stop inheritance at the branch.
    if stop_if_branch:
        remove = [node for node, degree in dict(
            traceback_graph.degree()).items() if degree > 2]
        traceback_graph.remove_nodes_from(remove)

    fragment_pIDs = nx.node_connected_component(
        traceback_graph.to_undirected(), tipID)

    return fragment_pIDs


def floodFillFromNode(depth_dict:  dict, max_filled: int) -> List[int]:
    """Given a dictionary where keys represent node IDs in a undirected graph,
    and values represent the distance of all nodes from a selected source, 
    run a flood fill of the network originating from the source node specified in 
    depth_dict until max_filled nodes are filled. Returns a list of node IDs that
    would be filled up to max_filled.
    
    depth_dict must contain ONE key: value pair where the key is a node ID, and the
    value is 0.  This is the source node to start flood filling from

    Args:
        depth_dict (dict): distance dictionary of all node distances to a specified 
            source node
        max_filled (int): depth to fill the network to

    Returns:
        List[int]: List of nodes that would be filled
    """
    filled = []
    max_fill_depth = len(depth_dict.keys())
    # If the network is smaller than that of the max fill desired, return the whole
    # network's IDs
    if max_fill_depth <= max_filled:
        return list(depth_dict.keys())

    for depth_val in range(max_fill_depth):
        nodes_at_depth = [k for k, v in depth_dict.items() if v == depth_val]
        # While the number of filled nodes isn't fully populated, and
        # while the pool of nodes at depth_val isn't exhausted yet,
        # sample nodes from list randomly when applicable
        # and populate filled array.
        while len(filled) < max_filled and len(nodes_at_depth) != 0:
            # Shuffle nodes if there is more than 1 for random ordering.
            if len(nodes_at_depth) > 1:
                shuffle(nodes_at_depth)
            filled += [nodes_at_depth.pop(0)]
        if len(filled) == max_filled:
            return filled
        
        
def recreate_graph_at_time_resolution(G:nx.DiGraph, resolution: float, 
                                      start_time: float=0, end_time: float=np.inf, precision: int = 1)-> nx.DiGraph:
    """
    Recreate the graph with nodes whose timestamp has a modulo of 0 with the given resolution.
    If a path of nodes is found that has timestamps not meeting the resolution condition, only
    the endpoints are added to the new graph.
    Graph nodes must have attribute "timestamp"

    Args:
        G (nx.DiGraph): The input graph.
        resolution (float): The time resolution.
        start_time (float): timestamp to start at; must be greater than or equal to zero
        end_time (float): timestamp to end at; defaults to np.inf
        precision (int): precision (number of decimal points) to round timestamp to, for correction of 
            floating point error noise. Defaults to 1.

    Returns:
        nx.DiGraph: The new graph with nodes whose timestamp has a modulo of 0 with the resolution.
    """
    
    if start_time <0:
        start_time = 0
        assert False,"Start time must be greater than or equal to 0"
        
    
    timestamp2nodes  ={}
    for nid, ndata in G.nodes(data=True):
        ts = np.around(ndata.get("timestamp"), decimals=precision)
        if np.around(ts, decimals=precision) % resolution==0  and start_time <= ts <= end_time:
            timestamp2nodes.setdefault(ts, []).append(nid) 
               
    if not timestamp2nodes:
        raise ValueError("Dictionary is empty")
    new_history_graph = nx.DiGraph()   
    
    min_time = min(list(timestamp2nodes.keys())+[start_time])
    if np.isinf(end_time):
        max_time =max(timestamp2nodes.keys())
    else:
        max_time = max(list(timestamp2nodes.keys())+[end_time])
    
    timestamp = min_time
    while (timestamp+resolution)<= max_time:
        
        nodes_at_pre_timestep = timestamp2nodes.get(timestamp)
        nodes_at_post_timestep = timestamp2nodes.get(timestamp + resolution)
        if not nodes_at_post_timestep:
            continue
        for pre_node_ID, post_node_ID in itertools.product(nodes_at_pre_timestep, nodes_at_post_timestep):
            if nx.has_path(G, pre_node_ID, post_node_ID):
                if not new_history_graph.has_node(pre_node_ID):
                    new_history_graph.add_node(pre_node_ID, **G.nodes[pre_node_ID])
                if not new_history_graph.has_node(post_node_ID):
                    new_history_graph.add_node(post_node_ID, **G.nodes[post_node_ID])
                new_history_graph.add_edge(pre_node_ID, post_node_ID)
        timestamp += resolution  
    return new_history_graph
