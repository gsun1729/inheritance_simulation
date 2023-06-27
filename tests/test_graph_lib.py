import networkx as nx
from lib.graph import (containsClique, containsCycle, floodFillFromNode, 
                       getInheritedChain, recreate_graph_at_time_resolution)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 1)
    print(containsCycle(G))
    print(containsClique(G))

    a = nx.Graph()
    a.add_edge(17, 1)
    a.add_edge(1, 2)
    a.add_edge(2, 3)
    a.add_edge(3, 4)
    a.add_edge(4, 5)
    a.add_edge(5, 6)
    a.add_edge(6, 7)
    a.add_edge(6, 8)
    a.add_edge(8, 9)
    a.add_edge(7, 10)
    a.add_edge(9, 11)
    a.add_edge(11, 12)
    a.add_edge(12, 13)
    a.add_edge(10, 14)
    a.add_edge(14, 15)
    a.add_edge(15, 16)
    a.add_edge(4, 18)
    a.add_edge(14, 19)
    a.add_edge(19, 20)
    a.add_edge(10, 20)
    a.add_node(21)

    # nx.draw(a, with_labels=True); plt.show()
    d = {1: 9, 2: 9, 3: 0, 4: 17, 5: 2, 6: 11, 7: 1, 8: 19, 10: 12, 13: 14, 14: 10, 16: 8, 17: 15, 22: 18, 25: 17, 26: 7, 27: 13, 28: 15, 29: 5,
         30: 15, 31: 13, 32: 11, 33: 19, 34: 3, 35: 18, 39: 20, 41: 10, 43: 16, 44: 5, 45: 6, 46: 14, 47: 8, 48: 4, 49: 10, 51: 10, 52: 16, 53: 6}
    f = floodFillFromNode(d, 15)
    print(f)
    getInheritedChain(a, 20, False)
    getInheritedChain(a, 20, True)


    G = nx.DiGraph()
    G.add_node(0, timestamp = 0)
    G.add_node(1, timestamp = 1)
    G.add_node(2, timestamp = 2)
    G.add_node(3, timestamp = 3)
    G.add_node(4, timestamp = 4)
    G.add_node(5, timestamp = 5)
    G.add_node(6, timestamp = 6)
    G.add_node(7, timestamp = 7)
    G.add_node(8, timestamp = 8)
    G.add_edge(0,1)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(3,4)
    G.add_edge(4,5)
    G.add_edge(5,6)
    G.add_edge(6,7)
    G.add_edge(7,8)

    G.add_node(9, timestamp=2)
    G.add_node(10, timestamp=3)
    G.add_node(11, timestamp=4)
    G.add_node(12, timestamp=5)

    G.add_node(13, timestamp=2)
    G.add_node(14, timestamp=3)
    G.add_node(15, timestamp=4)
    G.add_node(16, timestamp=5)


    G.add_edge(9,10)
    G.add_edge(10,11)
    G.add_edge(11,12)
    G.add_edge(1,9)
    G.add_edge(12,6)

    G.add_edge(13,14)
    G.add_edge(14,15)
    G.add_edge(15,16)
    G.add_edge(1,13)
    G.add_edge(16,6)

    pos = nx.nx_agraph.graphviz_layout(G,
                                    prog='dot',
                                    args="-Grankdir=LR")
    nx.draw(G, with_labels=True,
            pos=pos,
            node_color=["r" if ndata.get("timestamp")%2==0 else "g" for _, ndata in G.nodes(data=True)],
            connectionstyle="angle, angleA=-90,angleB=180, rad=0")
    new_history_graph = recreate_graph_at_time_resolution(G, 2,start_time=0,end_time=6)
    plt.show()

    pos = nx.nx_agraph.graphviz_layout(new_history_graph,
                                    prog='dot',
                                    args="-Grankdir=LR")
    nx.draw(new_history_graph, with_labels=True,
            pos=pos,
            node_color=["r" if ndata.get("timestamp")%2==0 else "g" for _, ndata in new_history_graph.nodes(data=True)],
            connectionstyle="angle, angleA=-90,angleB=180, rad=0")
    plt.show()
    for n, ndata in new_history_graph.nodes(data= True): 
        print(n, ndata)
        