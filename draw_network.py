import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')  # First column is row labels
    return df.index.tolist(), df.columns.tolist(), df.values  # Extract row labels, column labels, and matrix

#def sort(file_path):
    #df = pd.read_csv(file_path, encoding='ISO-8859-1')
    #df_sorted = df.sort_values(by=df.columns[2], ascending=True)
    #return df_sorted
    # Save the reordered DataFrame to a new CSV file
   # df_sorted.to_csv('restored_sorted.csv', index=False)

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
    for plant in enumerate(plants):
        abundance_value = floral_abundance.loc[floral_abundance.iloc[:, 0] == plant, floral_abundance.columns[7]].values
        size = abundance_value[0] if len(abundance_value) > 0 else 1  # Default size if not found
        G.add_node(plant, bipartite=0, size=size)

# Add animal nodes
    for pollinator in enumerate(pollinators):
        G.add_node(pollinator, bipartite=1)

    # Add nodes with bipartite attribute
    # G.add_nodes_from(pollinators, bipartite=0)  # Set 1 (plants)
    # G.add_nodes_from(plants, bipartite=1)  # Set 2 (pollinators)

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
    weights = [d['weight'] for _,_,d in edges]  # Extract weights+
    max_weight = max(weights) if weights else 1
    normalized_weights = [0.5 + (w / max_weight) * 5 for w in weights]
    
    
   # plant_sizes = dict(zip(file2.iloc[:, 2], file2.iloc[:, 7]))
   # plant_size_values = [plant_sizes.get(node, 1000) for node in G.nodes()]
   # node_sizes = [100] * len(pollinators) + plant_size_values
   
    sizes = [G.nodes[n]["size"] * 100 for n in G.nodes if "size" in G.nodes[n]]
    
    nx.draw(G, pos, with_labels=True, node_size=sizes, node_color="lightblue", edge_color="black", font_size=12, width=normalized_weights)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in edges}, font_size=10)
    
#controlled_sorted = sort("controlled.csv")
#controlled_sorted.to_csv('controlled_sorted.csv', index=False)

#sort("restored plants.csv")
print_network("grestored.csv", "restored_sorted.csv")
plt.title("Bipartite Pollination Graph Restored")
    
# Save the figure as PDF
plt.savefig("restored_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

#print_network("grestored.csv")
#plt.title("Bipartite Pollination Graph Restored")
    
# Save the figure as PDF
#plt.savefig("restored_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
#plt.show()