import pandas as pd
import numpy as np
import networkx as nx

# Load the adjacency matrix from the CSV file (assuming it has no header, and first row/column are node names)
adj_matrix = pd.read_csv('grestored.csv', index_col=0)

# Create an empty bipartite graph
B = nx.Graph()

# Add edges to the graph from the adjacency matrix
for i, row in adj_matrix.iterrows():
    for j, weight in row.iteritems():
        if weight > 0:  # Only add the edge if there is a non-zero weight
            B.add_edge(i, j, weight=weight)

# Function to calculate the Shannon entropy for a node based on its adjacency weights
def shannon_entropy(weights):
    total_weight = np.sum(weights)
    proportions = weights / total_weight  # Proportions of each edge weight
    entropy = -np.sum(proportions * np.log(proportions + 1e-9))  # Add a small value to avoid log(0)
    return entropy

# Function to calculate the Shannon Evenness Index for a node
def evenness(weights):
    weighted_degree = np.sum(weights)
    entropy = shannon_entropy(weights)
    evenness_index = entropy / np.log(weighted_degree) if weighted_degree > 1 else 0
    return evenness_index

# Identify nodes in set A (rows) and set B (columns)
A_nodes = list(adj_matrix.index)
B_nodes = list(adj_matrix.columns)

# Compute evenness for nodes in set A (rows in the adjacency matrix)
evenness_A = []
for a in A_nodes:
    weights = adj_matrix.loc[a, :].values  # Get the weights for node 'a' to all nodes in set B
    evenness_A.append(evenness(weights))

# Compute evenness for nodes in set B (columns in the adjacency matrix)
evenness_B = []
for b in B_nodes:
    weights = adj_matrix[b].values  # Get the weights for node 'b' to all nodes in set A
    evenness_B.append(evenness(weights))

# Print the results
print("Evenness for nodes in set A:", dict(zip(A_nodes, evenness_A)))
print("Evenness for nodes in set B:", dict(zip(B_nodes, evenness_B)))

# Calculate overall evenness of the network
overall_evenness = (np.mean(evenness_A) + np.mean(evenness_B)) / 2
print("Overall Evenness of the network:", overall_evenness)