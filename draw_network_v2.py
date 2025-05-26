import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')
    return df.index.tolist(), df.columns.tolist(), df.values

def print_network(file_path, file_path2, common_names):
    # Load adjacency matrix
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_path)
    floral_abundance = pd.read_csv(file_path2, encoding='ISO-8859-1')

    # Create bipartite graph
    G = nx.Graph()
    pollinators = row_labels
    plants = col_labels

    node_sizes = {}
    plant_sizes = []
    MIN_SIZE = 100  # Minimum size for plant nodes

    # Add plant nodes with size based on abundance
    for plant in plants:
        abundance_value = floral_abundance.loc[
            floral_abundance.iloc[:, 2] == plant,
            floral_abundance.columns[7]
        ].values
        size = float(abundance_value[0]) if len(abundance_value) > 0 else 1.0
        scaled_size = max(size * 3000, MIN_SIZE)
        G.add_node(plant, bipartite=0)
        node_sizes[plant] = scaled_size
        plant_sizes.append(scaled_size)

    # Add pollinator nodes with default size
    for pollinator in pollinators:
        G.add_node(pollinator, bipartite=1)
        node_sizes[pollinator] = 100

    # Add edges based on adjacency matrix
    for i, pollinator in enumerate(pollinators):
        for j, plant in enumerate(plants):
            if adj_matrix[i, j] > 0:
                G.add_edge(pollinator, plant, weight=adj_matrix[i, j])

    # --- PLANT POSITIONS: symmetrically spaced based on abundance ---
    total_size = sum(plant_sizes)
    normalized_sizes = [s / total_size for s in plant_sizes]
    min_spacing = 0.02

    centers = []
    y = 0
    for h in normalized_sizes:
        y += h / 2
        centers.append(y)
        y += h / 2 + min_spacing

    mid_y = (centers[0] + centers[-1]) / 2
    centered_positions = [c - mid_y for c in centers]

    pos = {}
    scale_factor = 100
    for plant, y in zip(plants, centered_positions):
        pos[plant] = (0, y * scale_factor)

    # --- POLLINATOR POSITIONS: evenly spaced across full plant range
    min_y = min(centered_positions) * scale_factor
    max_y = max(centered_positions) * scale_factor
    pollinator_y = np.linspace(min_y, max_y, len(pollinators))

    for pollinator, y in zip(pollinators, pollinator_y):
        pos[pollinator] = (1, y)

    # Determine node colors
    node_colors = [
        "blue" if G.nodes[node].get("bipartite") == 1 or node in common_names else "red"
        for node in G.nodes
    ]

    # Normalize edge weights for line width
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]
    max_weight = max(weights) if weights else 1
    normalized_weights = [0.5 + (w / max_weight) * 5 for w in weights]

    # Draw the graph
    plt.figure(figsize=(50, 32))
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=[node_sizes[n] for n in G.nodes],
        node_color=node_colors,
        edge_color="black",
        width=normalized_weights
    )

    # Draw labels manually — offset from nodes
    label_offset = 0.02
    for node, (x, y) in pos.items():
        if G.nodes[node]['bipartite'] == 0:  # plant
            plt.text(x - label_offset, y, node, ha='right', va='center', fontsize=20, rotation=0)
        else:  # pollinator
            plt.text(x + label_offset, y, node, ha='left', va='center', fontsize=20, rotation=0)

    # Optional: legend + title
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Other Plants / Pollinators'),
        mpatches.Patch(color='orange', label='Highlighted Plants')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Plant-Pollinator Bipartite Network", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()

    
def shannon_entropy(prob):
    entropy = -np.sum(prob * np.log2(prob + 1e-9))  # Add a small value to avoid log(0)
    return entropy

# falso
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

#evennesses = []
controlled_families = ["gcontrolled_coleoptera.csv", "gcontrolled_diptera.csv", "gcontrolled_hymenoptera.csv", "gcontrolled_lepidoptera.csv", "gcontrolled_squamata.csv"]
#for i in controlled_families:
#    append(evennesses, eveness(i))
#print(sum(evennesses)/len(evennesses))

pollinators_c, plants_c, adj_matrix_c = load_adjacency_matrix("gcontrolled.csv")
pollinators_r, plants_r, adj_matrix_r = load_adjacency_matrix("grestored.csv")

common_pollinators = set(pollinators_c) & set(pollinators_r)
common_plants = set(plants_c) & set(plants_r)

evenness_c = list(map(evenness, controlled_families))
print(evenness_c)
print(sum(evenness_c)/len(controlled_families))

evenness_r = evenness("grestored.csv")
evenness_c = evenness("gcontrolled.csv")
print("controlled evenness:", evenness_c)
print("restored evenness:", evenness_r)

    
print_network("gcontrolled.csv", "controlled_sorted.csv", common_plants)
plt.title("Bipartite Pollination Graph Restored")
    
# Save the figure as PDF
plt.savefig("controlled_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

