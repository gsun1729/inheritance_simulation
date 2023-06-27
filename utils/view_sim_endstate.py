"""Script used for visualizing simulation end state at end of run, given a filepath of a simulation pickle recording"""
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import sys

PATH = sys.argv[-1]

with open(PATH, 'rb') as f:
    post_case = pickle.load(f)
    
    
max_timestamp = max(post_case.network_history.keys())
max_network = post_case.network_history[max_timestamp]


c = ["g" if not p['has_agg'] else "r" for _, p in max_network.nodes(data=True)]
nx.draw(max_network, node_color=c,with_labels=True)
plt.show()
