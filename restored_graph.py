import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')  # First column is row labels
    return df.index.tolist(), df.columns.tolist(), df.values  # Extract row labels, column labels, and matrix

# Load adjacency matrix from CSV with headers
row_labels, col_labels, adj_matrix = load_adjacency_matrix("grestored.csv")

# Create a bipartite graph
G = nx.Graph()

# Define node sets
pollinators = row_labels  # Use actual plant names from CSV headers
plants = col_labels  # Use actual pollinator names from CSV headers

# Add nodes with bipartite attribute
G.add_nodes_from(pollinators, bipartite=0)  # Set 1 (plants)
G.add_nodes_from(plants, bipartite=1)  # Set 2 (pollinators)

# Add edges based on adjacency matrix
for i, pollinator in enumerate(pollinators):
    for j, plant in enumerate(plants):
        if adj_matrix[i, j] > 0:
            G.add_edge(pollinator, plant, weight=adj_matrix[i, j])

# Positioning the nodes
pos = nx.bipartite_layout(G, plants)

# Draw the graph
plt.figure(figsize=(30, 26))
edges = G.edges(data=True)
weights = [d['weight'] for _,_,d in edges]  # Extract weights

nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="black", font_size=12, width=weights)
#nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=10)

plt.title("Bipartite Pollination Graph")

# Save the figure as PDF
plt.savefig("bipartite_graph_restored.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()
