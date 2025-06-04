import networkx as nx
import graph_tool.all as gt
import numpy as np

# Assuming your networkx bipartite graph `G_nx` has node attributes
# indicating their partition (e.g., 'bipartite': 0 for plants, 'bipartite': 1 for pollinators)

# Example: Create a dummy bipartite networkx graph
G_nx = nx.Graph()
plants = ['P1', 'P2', 'P3']
pollinators = ['L1', 'L2', 'L3', 'L4']

G_nx.add_nodes_from(plants, bipartite=0)
G_nx.add_nodes_from(pollinators, bipartite=1)

G_nx.add_edges_from([('P1', 'L1'), ('P1', 'L2'), ('P2', 'L2'),
                     ('P2', 'L3'), ('P3', 'L3'), ('P3', 'L4')])

# Convert NetworkX graph to graph-tool graph
g = gt.Graph(directed=False) # Your plant-pollinator network is likely undirected

# Add vertices and their bipartite property
node_map_nx_to_gt = {}
bipartite_prop = g.new_vertex_property("int") # 0 for plants, 1 for pollinators

for i, node_id in enumerate(G_nx.nodes()):
    v = g.add_vertex()
    node_map_nx_to_gt[node_id] = v
    bipartite_prop[v] = G_nx.nodes[node_id]['bipartite']

g.vp.bipartite = bipartite_prop

# Add edges
for u_nx, v_nx in G_nx.edges():
    u_gt = node_map_nx_to_gt[u_nx]
    v_gt = node_map_nx_to_gt[v_nx]
    g.add_edge(u_gt, v_gt)

print("Graph-tool graph created with bipartite property.")