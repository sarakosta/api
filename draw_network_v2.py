import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')  # First column is row labels
    return df.index.tolist(), df.columns.tolist(), df.values  # Extract row labels, column labels, and matrix

def print_network(file_path, file_path2):
    # Load adjacency matrix from CSV with headers
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_path)

    # Create a bipartite graph
    G = nx.Graph()

    # Define node sets
    pollinators = row_labels  # Use actual plant names from CSV headers
    plants = col_labels  # Use actual pollinator names from CSV headers
    
    floral_abundance = pd.read_csv(file_path2, encoding='ISO-8859-1')
    
    # Add plant nodes
    node_sizes = {}
    for _, plant in enumerate(plants):
        abundance_value = floral_abundance.loc[floral_abundance.iloc[:, 2] == plant, floral_abundance.columns[7]].values
        size = float(abundance_value[0]) if len(abundance_value) > 0 else 1.0  # Default size if not found
        G.add_node(plant, bipartite=0)
        node_sizes[plant] = size * 1000

    # Add animal nodes
    for _, pollinator in enumerate(pollinators):
        G.add_node(pollinator, bipartite=1)
        node_sizes[pollinator] = 100  # Default size for animal nodes

    # Add edges based on adjacency matrix
    for i, pollinator in enumerate(pollinators):
        for j, plant in enumerate(plants):
            if adj_matrix[i, j] > 0:
                G.add_edge(pollinator, plant, weight=adj_matrix[i, j])

    # Positioning the nodes
    pos = nx.bipartite_layout(G, plants)

    # Ensure node sizes match positions
    sizes = [node_sizes[n] for n in G.nodes]
    
    # Draw the graph
    plt.figure(figsize=(30, 26))
    edges = G.edges(data=True)
    weights = [d['weight'] for _,_,d in edges]  # Extract weights
    max_weight = max(weights) if weights else 1
    normalized_weights = [0.5 + (w / max_weight) * 5 for w in weights]
    
    nx.draw(G, pos, with_labels=True, node_size=sizes, node_color="lightblue", edge_color="black", font_size=12, width=normalized_weights)
    
def shannon_entropy(prob):
    entropy = -np.sum(prob * np.log2(prob + 1e-9))  # Add a small value to avoid log(0)
    return entropy

def evenness0(file_path):
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_path)
    plant_weights =  [np.sum(col) for col in adj_matrix.T]
    animal_weights = [np.sum(row) for row in adj_matrix]
    prob_plant = np.zeros((len(animal_weights), len(plant_weights)))
    prob_animal = np.zeros((len(animal_weights), len(plant_weights)))
    for i in range(len(animal_weights)):
        for j in range(len(plant_weights)):
            if plant_weights[j] != 0:  # Check for division by zero
                prob_plant[i, j] = adj_matrix[i, j] / plant_weights[j]
            else:
                prob_plant[i, j] = 0  # or np.nan
            if animal_weights[j] != 0:  # Check for division by zero
                prob_animal[i, j] = adj_matrix[i, j] / animal_weights[i]
            else:
                prob_animal[i, j] = 0  # or np.nan
    evenness_animal = shannon_entropy(prob_animal) / np.log2(np.sum(adj_matrix))
    evenness_plant = shannon_entropy(prob_plant) / np.log2(np.sum(adj_matrix))
    return evenness_plant, evenness_animal

def evenness(file_path):
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_path)
    prob = adj_matrix/np.sum(adj_matrix)
    evenness = shannon_entropy(prob) / np.log2(np.count_nonzero(adj_matrix))
    return evenness

evenness_r = evenness("grestored.csv")
evenness_c = evenness("gcontrolled.csv")
print("controlled evenness:", evenness_c)
print("restored evenness:", evenness_r)

    
print_network("gcontrolled.csv", "controlled_sorted.csv")
plt.title("Bipartite Pollination Graph Restored")
    
# Save the figure as PDF
plt.savefig("controlled_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

