import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')
    return df.index.tolist(), df.columns.tolist(), df.values

def print_network_order(file_adj_matrix, file_plants, file_animals, common_names):

    # Load data
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_adj_matrix)
    floral_abundance = pd.read_csv(file_plants, encoding='ISO-8859-1')
    animal_df = pd.read_csv(file_animals, encoding='ISO-8859-1')

    # Build species-to-order mapping
    species_to_order = {
        species.strip(): order.strip()
        for species, order in zip(animal_df.iloc[:, 3], animal_df.iloc[:, 1])
    }

    # Define custom colors for each order
    order_color_map = {
        "Diptera": "skyblue",
        "Hymenoptera": "orange",
        "Coleoptera": "green",
        "Lepidoptera": "purple",
        "Hemiptera": "gray",
        "Passeriformes": "brown",
        "Squamata": "pink",
    }
    default_color = "black"

    # Check for species in the matrix not found in the animal file
    missing_species = [s for s in row_labels if s.strip() not in species_to_order]
    if missing_species:
        print("⚠️ Species in network not found in animal CSV:")
        for s in missing_species:
            print(f" - '{s}'")

    # Create bipartite graph
    G = nx.Graph()
    pollinators = row_labels
    plants = col_labels

    node_sizes = {}
    plant_sizes = []
    pollinator_sizes = []
    MIN_SIZE = 100

    # Add plant nodes
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

    for pollinator in pollinators:
        n_visits = animal_df.loc[
            animal_df.iloc[:, 3] == pollinator,
            animal_df.columns[6]
        ].values
        size = float(n_visits[0]) if len(n_visits) > 0 else 1.0
        scaled_size = max(size * 10, MIN_SIZE)
        G.add_node(pollinator, bipartite=1)
        node_sizes[pollinator] = scaled_size
        pollinator_sizes.append(scaled_size)

    # Add edges
    for i, pollinator in enumerate(pollinators):
        for j, plant in enumerate(plants):
            if adj_matrix[i, j] > 0:
                G.add_edge(pollinator, plant, weight=adj_matrix[i, j])

    # Plant positions
    total_size_pl = sum(plant_sizes)
    normalized_sizes_pl = [s / total_size_pl for s in plant_sizes]
    min_spacing = 0.02
    centers = []
    y = 0
    for h in normalized_sizes_pl:
        y += h / 2
        centers.append(y)
        y += h / 2 + min_spacing
    mid_y = (centers[0] + centers[-1]) / 2
    centered_positions = [c - mid_y for c in centers]
    plant_pos = {}
    x_plant = 0
    scale_factor = 100
    for plant, y in zip(plants, centered_positions):
        plant_pos[plant] = (x_plant, y * scale_factor)

    # pollinator positions
    total_size_pol = sum(pollinator_sizes)
    normalized_sizes_pol = [s / total_size_pol for s in pollinator_sizes]
    min_spacing = 0.02
    centers = []
    y = 0
    for h in normalized_sizes_pol:
        y += h / 2
        centers.append(y)
        y += h / 2 + min_spacing
    mid_y = (centers[0] + centers[-1]) / 2
    centered_positions = [c - mid_y for c in centers]
    pollinator_pos = {}
    x_pollinator = 1
    for pollinator, y in zip(pollinators, centered_positions):
        pollinator_pos[pollinator] = (x_pollinator, y * scale_factor)
        
    # Merge into a single position dictionary
    pos = {**plant_pos, **pollinator_pos}

    # Node colors
    node_colors = []
    for node in G.nodes:
        if G.nodes[node]['bipartite'] == 0:
            color = "blue" if node in common_names else "red"
        else:
            order = species_to_order.get(node.strip())
            color = order_color_map.get(order, default_color)
        node_colors.append(color)

    # Edge weights
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]
    max_weight = max(weights) if weights else 1
    normalized_weights = [0.5 + (w / max_weight) * 5 for w in weights]

    # Plotting
    plt.figure(figsize=(50, 32))
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=[node_sizes[n] for n in G.nodes],
        node_color=node_colors,
        edge_color="black",
        width=normalized_weights
    )

    # Labels
    label_offset = 0.02
    for node, (x, y) in pos.items():
        if G.nodes[node]['bipartite'] == 0:
            plt.text(x - label_offset, y, node, ha='right', va='center', fontsize=20)
        else:
            plt.text(x + label_offset, y, node, ha='left', va='center', fontsize=20)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=o) for o, c in order_color_map.items()]
    legend_patches.append(mpatches.Patch(color="orange", label="Highlighted Plants"))
    legend_patches.append(mpatches.Patch(color="red", label="Other Plants"))
    legend_patches.append(mpatches.Patch(color="gray", label="Unknown Order"))

    plt.legend(handles=legend_patches, loc='upper right', fontsize=14)
    plt.title("Plant-Pollinator Bipartite Network", fontsize=16)
    plt.axis('off')
    # plt.tight_layout()
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

# draw an histogram
def histo_old(plant_linkage, pollinator_linkage):
    # Plot histogram
    plant_min_val = int(np.floor(plant_linkage.min()))
    plant_max_val = int(np.ceil(plant_linkage.max()))
    bins_plant = np.arange(plant_min_val - 0.5, plant_max_val + 1.5, 1)
    
    pollinator_min_val = int(np.floor(pollinator_linkage.min()))
    pollinator_max_val = int(np.ceil(pollinator_linkage.max()))
    bins_pollinator = np.arange(pollinator_min_val - 0.5, pollinator_max_val + 1.5, 1) 
    
    plt.figure(figsize=(8, 5))
    plt.hist(plant_linkage, bins=bins_plant, align='left', edgecolor='black')
    plt.title('Linkage Distribution for Plants')
    plt.xlabel('Plant Degree (number of interactions)')
    plt.ylabel('Number of plant nodes')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.hist(pollinator_linkage, bins=bins_pollinator, align='left', edgecolor='black')
    plt.title('Linkage Distribution for Pollinators')
    plt.xlabel('Pollinator Degree (number of interactions)')
    plt.ylabel('Number of pollinator nodes')
    plt.grid(True)
    plt.show()
    
def histo_side_by_side(
    plant_linkage_controlled, plant_linkage_restored,
    pollinator_linkage_controlled, pollinator_linkage_restored
):
    # --- Plants ---
    all_plant = np.concatenate([plant_linkage_controlled, plant_linkage_restored])
    plant_min_val = int(np.floor(all_plant.min()))
    plant_max_val = int(np.ceil(all_plant.max()))
    bins_plant = np.arange(plant_min_val - 0.5, plant_max_val + 1.5, 1)

    plt.figure(figsize=(8, 5))
    plt.hist(
        [plant_linkage_controlled, plant_linkage_restored],
        bins=bins_plant,
        label=['Controlled', 'Restored'],
        align='left',
        edgecolor='black',
        alpha=0.7
    )
    plt.title('Linkage Distribution for Plants')
    plt.xlabel('Plant Degree (number of interactions)')
    plt.ylabel('Number of plant nodes')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Pollinators ---
    all_pollinators = np.concatenate([pollinator_linkage_controlled, pollinator_linkage_restored])
    pollinator_min_val = int(np.floor(all_pollinators.min()))
    pollinator_max_val = int(np.ceil(all_pollinators.max()))
    bins_pollinator = np.arange(pollinator_min_val - 0.5, pollinator_max_val + 1.5, 1)

    plt.figure(figsize=(8, 5))
    plt.hist(
        [pollinator_linkage_controlled, pollinator_linkage_restored],
        bins=bins_pollinator,
        label=['Controlled', 'Restored'],
        align='left',
        edgecolor='black',
        alpha=0.7
    )
    plt.title('Linkage Distribution for Pollinators')
    plt.xlabel('Pollinator Degree (number of interactions)')
    plt.ylabel('Number of pollinator nodes')
    plt.legend()
    plt.grid(True)
    plt.show()


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

    
print_network_order("gcontrolled.csv", "controlled_sorted.csv", "controlled_animal_sorted.csv" ,common_plants)
plt.title("Bipartite Pollination Graph Controlled")    

# Save the figure as PDF
plt.savefig("controlled_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

print_network_order("grestored.csv", "restored_sorted.csv", "restored_animal_sorted.csv" ,common_plants)
plt.title("Bipartite Pollination Graph Restored")    

# Save the figure as PDF
plt.savefig("restored_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

plant_info_c = pd.read_csv("controlled_sorted.csv", encoding='ISO-8859-1')
pollinator_info_c = pd.read_csv("controlled_animal_sorted.csv", encoding='ISO-8859-1')

plant_linkage_c = plant_info_c.iloc[:, 8]
plant_linkage_c = pd.to_numeric(plant_linkage_c, errors='coerce').dropna()

pollinator_linkage_c = pollinator_info_c.iloc[:, 8]
pollinator_linkage_c = pd.to_numeric(pollinator_linkage_c, errors='coerce').dropna()

plant_info_r = pd.read_csv("restored_sorted.csv", encoding='ISO-8859-1')
pollinator_info_r = pd.read_csv("restored_animal_sorted.csv", encoding='ISO-8859-1')

plant_linkage_r = plant_info_r.iloc[:, 8]
plant_linkage_r = pd.to_numeric(plant_linkage_r, errors='coerce').dropna()

pollinator_linkage_r = pollinator_info_r.iloc[:, 8]
pollinator_linkage_r = pd.to_numeric(pollinator_linkage_r, errors='coerce').dropna()

histo_side_by_side(plant_linkage_c, plant_linkage_r, pollinator_linkage_c, pollinator_linkage_r)

# get the degree given the csv file (it is the same as the linkage!!!!)
def degree(csv_path):
    # Load the weighted adjacency matrix with headers and row names
    df = pd.read_csv(csv_path, index_col=0)
    
    # Convert to binary (presence/absence)
    binary_df = (df > 0).astype(int)

    # Compute degrees
    plant_degrees = binary_df.sum(axis=1)         # sum across columns
    pollinator_degrees = binary_df.sum(axis=0)    # sum across rows
    
    all_degrees = pd.concat([plant_degrees, pollinator_degrees])

    # Return Series with labels
    return plant_degrees, pollinator_degrees, all_degrees

plant_degrees_c, pollinator_degrees_c, all_degrees_c = degree("gcontrolled.csv")
plant_degrees_r, pollinator_degrees_r, all_degrees_r = degree("grestored.csv")



def classify_by_degree(degree_series, specialists_path, generalists_path, low_thresh=1, high_thresh=99):
    """
    Classify species by degree and save specialists and generalists in separate CSV files.

    Parameters:
    - degree_series: pd.Series with species names as index and degrees as values
    - specialists_path: filepath to save specialists CSV
    - generalists_path: filepath to save generalists CSV
    - low_thresh: percentile cutoff for specialists (default 25)
    - high_thresh: percentile cutoff for generalists (default 75)

    Returns:
    - classification: pd.Series with all classifications
    """
    import numpy as np

    low = np.percentile(degree_series, low_thresh)
    high = np.percentile(degree_series, high_thresh)

    def classify(k):
        if k <= low:
            return "specialist"
        elif k >= high:
            return "generalist"
        else:
            return "intermediate"

    classification = degree_series.apply(classify)

    # Save specialists
    specialists = classification[classification == "specialist"].reset_index()
    specialists.columns = ['species', 'classification']
    specialists.to_csv(specialists_path, index=False)

    # Save generalists
    generalists = classification[classification == "generalist"].reset_index()
    generalists.columns = ['species', 'classification']
    generalists.to_csv(generalists_path, index=False)

    return classification


classification_c = classify_by_degree(all_degrees_c, "specialists_controlled.txt", "generalists_controlled.txt")
classification_r = classify_by_degree(all_degrees_r, "specialists_restored.txt", "generalists_restored.txt")
