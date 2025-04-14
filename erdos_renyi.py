import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')  # First column is row labels
    return df.index.tolist(), df.columns.tolist(), df.values  # Extract row labels, column labels, and matrix

def network(file_path):
    # Load adjacency matrix from CSV with headers
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_path)

    # Create a bipartite graph
    G = nx.Graph()

    # Define node sets
    pollinators = row_labels  # Use actual plant names from CSV headers
    plants = col_labels  # Use actual pollinator names from CSV headers
    
    # Add plant nodes
    for _, plant in enumerate(plants):
        G.add_node(plant, bipartite=0)

    # Add animal nodes
    for _, pollinator in enumerate(pollinators):
        G.add_node(pollinator, bipartite=1)

    # Add edges based on adjacency matrix
    for i, pollinator in enumerate(pollinators):
        for j, plant in enumerate(plants):
            if adj_matrix[i, j] > 0:
                G.add_edge(pollinator, plant, weight=adj_matrix[i, j])

    return G

def erdos_renyi(file_path):
    rows, cols, adj_matrix = load_adjacency_matrix(file_path)

    # Set numbers
    num_plants = len(cols)
    num_pollinators = len(rows)
    interactions = np.count_nonzero(adj_matrix)
    p = interactions / (num_plants * num_pollinators)  # Probability of interaction

    # Create bipartite graph
    B = nx.Graph()
    B.add_nodes_from(range(num_plants), bipartite=0)         # Plants
    B.add_nodes_from(range(num_plants, num_plants + num_pollinators), bipartite=1)  # Pollinators

    # Randomly add edges
    for i in range(num_plants):
        for j in range(num_pollinators):
            if random.random() < p:
                B.add_edge(i, num_plants + j)
    
    return B  

def degree(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    # Get degrees of all nodes
    plant_degrees = [graph.degree(p) for p in plants]
    pollinator_degrees = [graph.degree(p) for p in pollinators]
    
    return plant_degrees, pollinator_degrees


def histo(plant_degrees, pollinator_degrees):

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(plant_degrees, bins=range(0, max(plant_degrees)+2), align='left', edgecolor='black')
    plt.title('Degree Distribution of ER Bipartite Graph for Plants')
    plt.xlabel('Plant Degree (number of interactions)')
    plt.ylabel('Number of plant nodes')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.hist(pollinator_degrees, bins=range(0, max(pollinator_degrees)+2), align='left', edgecolor='black')
    plt.title('Degree Distribution of ER Bipartite Graph for Pollinators')
    plt.xlabel('Pollinator Degree (number of interactions)')
    plt.ylabel('Number of pollinator nodes')
    plt.grid(True)
    plt.show()
    
erdos_renyi = erdos_renyi("grestored.csv")
#plant_d_er, pollinator_d_er = degree(erdos_renyi)

restored_graph = network("grestored.csv")
#plant_d_api, pollinator_d_api = degree(restored_graph)

#histo(plant_d_er, pollinator_d_er) 
#histo(plant_d_api, pollinator_d_api)

def weighted_degree(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    # Get degrees of all nodes
    plant_w_degrees = [graph.degree(p, weight='weight') for p in plants]
    pollinator_w_degrees = [graph.degree(p, weight='weight') for p in pollinators]
    
    return plant_w_degrees, pollinator_w_degrees

plant_w_degree, pollinator_w_degree = weighted_degree(restored_graph)    
print(np.mean(plant_w_degree))
print(np.mean(pollinator_w_degree))

# centrality measures
def betweenness_centrality(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}
    
    betweenness = nx.betweenness_centrality(graph, weight = 'weight')
    plant_bcw = {n: betweenness[n] for n in plants}
    pollinator_bcw = {n: betweenness[n] for n in pollinators}
    
    return plant_bcw, pollinator_bcw

plant_bw , pollinator_bw = betweenness_centrality(restored_graph)

# Extract node names and values
nodes_plant = list(plant_bw.keys())
values_plant = list(plant_bw.values())
# Plot
plt.figure(figsize=(12, 6))
plt.bar(nodes_plant, values_plant, color='lightcoral', edgecolor='black')
plt.xticks(rotation=90)  # Rotate x labels for readability
plt.xlabel("Nodes")
plt.ylabel("Betweenness Centrality")
plt.title("Betweenness Centrality per Plant Nodes")
plt.tight_layout()  # Fix layout for better spacing
plt.show()

# Extract node names and values
nodes_pollinator = list(pollinator_bw.keys())
values_pollinator = list(pollinator_bw.values())
# Plot
plt.figure(figsize=(12, 6))
plt.bar(nodes_pollinator, values_pollinator, color='lightcoral', edgecolor='black')
plt.xticks(rotation=90)  # Rotate x labels for readability
plt.xlabel("Nodes")
plt.ylabel("Betweenness Centrality")
plt.title("Betweenness Centrality per Pollinator Nodes")
plt.tight_layout()  # Fix layout for better spacing
plt.show()

def closeness_centrality(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}
    
    closeness = nx.closeness_centrality(graph, distance = 'weight')
    plant_ccw = {n: closeness[n] for n in plants}
    pollinator_ccw = {n: closeness[n] for n in pollinators}
    
    return plant_ccw, pollinator_ccw

plant_cw , pollinator_cw = closeness_centrality(restored_graph)

# Extract node names and values
nodes_plant = list(plant_cw.keys())
values_plant = list(plant_cw.values())
# Plot
plt.figure(figsize=(12, 6))
plt.bar(nodes_plant, values_plant, color='lightcoral', edgecolor='black')
plt.xticks(rotation=90)  # Rotate x labels for readability
plt.xlabel("Nodes")
plt.ylabel("Closeness Centrality")
plt.title("Closeness Centrality per Plant Nodes")
plt.tight_layout()  # Fix layout for better spacing
plt.show()

# Extract node names and values
nodes_pollinator = list(pollinator_cw.keys())
values_pollinator = list(pollinator_cw.values())
# Plot
plt.figure(figsize=(12, 6))
plt.bar(nodes_pollinator, values_pollinator, color='lightcoral', edgecolor='black')
plt.xticks(rotation=90)  # Rotate x labels for readability
plt.xlabel("Nodes")
plt.ylabel("Closeness Centrality")
plt.title("Closeness Centrality per Pollinator Nodes")
plt.tight_layout()  # Fix layout for better spacing
plt.show()
