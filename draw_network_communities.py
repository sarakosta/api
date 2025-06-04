import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import community as co
import matplotlib.cm as cm
import graph_tool.all as gt

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')
    return df.index.tolist(), df.columns.tolist(), df.values

def shannon_entropy(prob):
    entropy = -np.sum(prob * np.log2(prob + 1e-9))
    return entropy

def evenness(file_path):
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_path)
    prob = adj_matrix / np.sum(adj_matrix)
    evenness_val = shannon_entropy(prob) / np.log2(np.count_nonzero(adj_matrix))
    return evenness_val

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
    plt.figure(figsize=(10, 10))
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

def print_network_communities(file_adj_matrix, file_plants, file_animals, common_names, network_type , output_txt_file_suffix="_community_details.txt"):
    # Load data
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_adj_matrix)
    floral_abundance = pd.read_csv(file_plants, encoding='ISO-8859-1')
    animal_df = pd.read_csv(file_animals, encoding='ISO-8859-1')

    # Build species-to-order mapping
    species_to_order = {
        species.strip(): order.strip()
        for species, order in zip(animal_df.iloc[:, 3], animal_df.iloc[:, 1])
    }

    # Define custom colors for each order (for general visualization, not community colors)
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
        G.add_node(plant, bipartite=0) # bipartite=0 for plants
        node_sizes[plant] = scaled_size
        plant_sizes.append(scaled_size)

    for pollinator in pollinators:
        n_visits = animal_df.loc[
            animal_df.iloc[:, 3] == pollinator,
            animal_df.columns[6]
        ].values
        size = float(n_visits[0]) if len(n_visits) > 0 else 1.0
        scaled_size = max(size * 10, MIN_SIZE)
        G.add_node(pollinator, bipartite=1) # bipartite=1 for pollinators
        node_sizes[pollinator] = scaled_size
        pollinator_sizes.append(scaled_size)

    # Add edges
    for i, pollinator in enumerate(pollinators):
        for j, plant in enumerate(plants):
            if adj_matrix[i, j] > 0:
                G.add_edge(pollinator, plant, weight=adj_matrix[i, j])

    print(f"\nNetwork loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Create graph-tool graph from NetworkX graph ---
    print("Converting NetworkX graph to graph-tool graph...")
    g_gt = gt.Graph(directed=False)

    # Create vertex property maps for node name, bipartite partition, and weight
    gt_name_prop = g_gt.new_vertex_property("string")
    gt_bipartite_prop = g_gt.new_vertex_property("int")
    gt_weight_prop = g_gt.new_edge_property("double") # For edge weights

    # Map NetworkX node names to graph-tool vertex objects
    gt_node_map = {}
    for node_name in G.nodes():
        v = g_gt.add_vertex()
        gt_node_map[node_name] = v
        gt_name_prop[v] = node_name
        gt_bipartite_prop[v] = G.nodes[node_name]['bipartite']

    # Add edges and their weights
    for u_nx, v_nx, data in G.edges(data=True):
        u_gt = gt_node_map[u_nx]
        v_gt = gt_node_map[v_nx]
        e = g_gt.add_edge(u_gt, v_gt)
        gt_weight_prop[e] = data.get('weight', 1.0) # Use 1.0 as default weight if not present

    g_gt.vp.name = gt_name_prop
    g_gt.vp.bipartite = gt_bipartite_prop
    g_gt.ep.weight = gt_weight_prop

    print(f"Graph-tool graph created with {g_gt.num_vertices()} vertices and {g_gt.num_edges()} edges.")
    # --- graph-tool graph creation complete ---

    # --- COMMUNITY DETECTION START ---
    print("\n--- Performing Community Detection ---")
    partition = co.best_partition(G, weight='weight')
    num_communities = max(partition.values()) + 1
    modularity = co.modularity(partition, G, weight='weight')

    print(f"Detected {num_communities} communities.")
    print(f"Modularity of the partition: {modularity:.4f}")

    # Prepare node colors based on community
    community_colors = cm.get_cmap('tab20', num_communities)
    node_community_colors = [community_colors(partition[node]) for node in G.nodes()]

    # --- Print Community Details to console AND file ---
    communities_grouped = {i: [] for i in range(num_communities)}
    for node, comm_id in partition.items():
        communities_grouped[comm_id].append(node)

    full_output_txt_filename = f"{network_type}{output_txt_file_suffix}" # Construct full filename
    with open(full_output_txt_filename, 'w', encoding='utf-8') as f:
        def print_and_write(text):
            print(text)
            f.write(text + '\n')
        print_and_write(f"\nNetwork loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. ({network_type} network)") # Added network_type for clarity in output text
        print_and_write("\n--- Performing Community Detection ---")
        print_and_write(f"Detected {num_communities} communities.")
        print_and_write(f"Modularity of the partition: {modularity:.4f}")
        print_and_write("\n--- Community Details ---")

        missing_species = [s for s in row_labels if s.strip() not in species_to_order]
        if missing_species:
            print_and_write("⚠️ Species in network not found in animal CSV:")
            for s in missing_species:
                print_and_write(f" - '{s}'")
            print_and_write("-" * 30)

        for comm_id, nodes_in_comm in communities_grouped.items():
            plants_in_comm = [n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 0]
            pollinators_in_comm = [n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 1]
            print_and_write(f"Community {comm_id}:")
            print_and_write(f"  Plants ({len(plants_in_comm)}): {', '.join(plants_in_comm) if plants_in_comm else 'None'}")
            print_and_write(f"  Pollinators ({len(pollinators_in_comm)}): {', '.join(pollinators_in_comm) if pollinators_in_comm else 'None'}")
            print_and_write("-" * 30)

    # --- COMMUNITY DETECTION END ---

    # --- POSITIONING (Original logic preserved) ---
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

    pos = {**plant_pos, **pollinator_pos}

    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]
    max_weight = max(weights) if weights else 1
    normalized_weights = [0.5 + (w / max_weight) * 5 for w in weights]

    # --- Plotting the main network graph ---
    plt.figure(figsize=(50, 32))
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=[node_sizes[n] for n in G.nodes],
        node_color=node_community_colors,
        edge_color="black",
        width=normalized_weights
    )

    label_offset = 0.02
    for node, (x, y) in pos.items():
        if G.nodes[node]['bipartite'] == 0:
            plt.text(x - label_offset, y, node, ha='right', va='center', fontsize=20)
        else:
            plt.text(x + label_offset, y, node, ha='left', va='center', fontsize=20)

    community_patches = []
    for i in range(num_communities):
        community_patches.append(mpatches.Patch(color=community_colors(i), label=f'Community {i}'))

    order_patches = [mpatches.Patch(color=c, label=o) for o, c in order_color_map.items()]
    order_patches.append(mpatches.Patch(color=default_color, label="Unknown Order"))

    plant_legend_patches = [
        mpatches.Patch(color="red", label="Other Plants"),
        mpatches.Patch(color="blue", label="Common Plants")
    ]

    all_legend_handles = community_patches + plant_legend_patches + order_patches
    plt.legend(handles=all_legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.title("Plant-Pollinator Bipartite Network with Detected Communities", fontsize=16)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # plt.show()

    # --- DRAWING INDIVIDUAL GRAPHS AND CREATING ADJACENCY MATRICES FOR EACH COMMUNITY ---
    print("\n--- Processing Individual Community Graphs and Adjacency Matrices ---")
    for comm_id, nodes_in_comm in communities_grouped.items():
        # Filter nodes into plants and pollinators within this community
        plants_in_comm = [n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 0]
        pollinators_in_comm = [n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 1]

        # Create a subgraph containing only nodes from the current community
        subgraph = G.subgraph(nodes_in_comm)

        # Skip if the community is empty or has no relevant edges for an adjacency matrix
        if not plants_in_comm or not pollinators_in_comm or subgraph.number_of_edges() == 0:
            if not nodes_in_comm:
                print(f"Community {comm_id} is empty. Skipping plot and CSV.")
            elif subgraph.number_of_edges() == 0:
                print(f"Community {comm_id} has nodes but no internal edges. Skipping plot and CSV.")
            elif not plants_in_comm:
                print(f"Community {comm_id} has no plants. Skipping CSV.")
            elif not pollinators_in_comm:
                print(f"Community {comm_id} has no pollinators. Skipping CSV.")
            continue # Skip to next community if no valid bipartite interactions

        # --- Create Adjacency Matrix for this Community ---
        # Initialize an empty DataFrame for the community's adjacency matrix
        # Pollinators as rows (index), Plants as columns
        community_adj_df = pd.DataFrame(
            0,
            index=pollinators_in_comm,
            columns=plants_in_comm
        )

        # Populate the DataFrame with edge weights from the subgraph
        for u, v, data in subgraph.edges(data=True):
            weight = data.get('weight', 0) # Get weight, default to 0 if not present
            # Determine which is pollinator and which is plant
            if G.nodes[u]['bipartite'] == 1 and G.nodes[v]['bipartite'] == 0: # u is pollinator, v is plant
                community_adj_df.loc[u, v] = weight
            elif G.nodes[u]['bipartite'] == 0 and G.nodes[v]['bipartite'] == 1: # u is plant, v is pollinator
                community_adj_df.loc[v, u] = weight # Always pollinator (row) to plant (column)

        # Save the community's adjacency matrix to a CSV file
        csv_filename = f"{network_type}_community_{comm_id}_adjacency_matrix.csv"
        community_adj_df.to_csv(csv_filename, encoding='utf-8')
        print(f"Saved {csv_filename}")
        print_network_order(f"{network_type}_community_{comm_id}_adjacency_matrix.csv", file_plants, file_animals, common_names)
        
        # --- Plotting the individual community graph (as before) ---
        plt.title(f"Community {comm_id} Network", fontsize=18)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{network_type}_community_{comm_id}_adjacency_matrix.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
        plt.close() # Keep this to avoid too many plot windows if you have many communities

    print("Finished processing individual community graphs and adjacency matrices.")
    
    return G, g_gt, partition 


# --- Main execution part (outside the function) ---
if __name__ == "__main__":
    # Ensure your data files exist for this to run
    # gcontrolled.csv, controlled_sorted.csv, controlled_animal_sorted.csv, grestored.csv
    # and the specific controlled_families CSVs (e.g., gcontrolled_coleoptera.csv etc.)

    pollinators_c, plants_c, adj_matrix_c = load_adjacency_matrix("gcontrolled.csv")
    pollinators_r, plants_r, adj_matrix_r = load_adjacency_matrix("grestored.csv")

    common_pollinators = set(pollinators_c) & set(pollinators_r)
    common_plants = set(plants_c) & set(plants_r)

    controlled_families = ["gcontrolled_coleoptera.csv", "gcontrolled_diptera.csv", "gcontrolled_hymenoptera.csv", "gcontrolled_lepidoptera.csv", "gcontrolled_squamata.csv"]
    evenness_c_list = []
    for f in controlled_families:
        try:
            evenness_c_list.append(evenness(f))
        except Exception as e:
            print(f"Could not calculate evenness for {f}: {e}")
            evenness_c_list.append(np.nan)

    if evenness_c_list and not all(np.isnan(evenness_c_list)):
        print(evenness_c_list)
        print(f"Average evenness for controlled families: {np.nanmean(evenness_c_list):.4f}")
    else:
        print("No valid evenness values calculated for controlled families.")

    try:
        evenness_r = evenness("grestored.csv")
        print("Restored evenness:", evenness_r)
    except Exception as e:
        print(f"Could not calculate evenness for grestored.csv: {e}")
        evenness_r = np.nan

    try:
        evenness_c = evenness("gcontrolled.csv")
        print("Controlled evenness:", evenness_c)
    except Exception as e:
        print(f"Could not calculate evenness for gcontrolled.csv: {e}")
        evenness_c = np.nan

    # Call the modified print_network function
    # The output will now be saved to 'controlled_community_details.txt'
    # And individual community adjacency matrices will be saved as CSVs
    print_network_communities("gcontrolled.csv", "controlled_sorted.csv", "controlled_animal_sorted.csv", common_plants, network_type = "controlled", output_txt_file_suffix="_community_details.txt")

    plt.title("Bipartite Pollination Graph - Controlled Network with Communities", fontsize=16)
    plt.savefig("controlled_graph_with_communities.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.show()
    
    print_network_communities("grestored.csv", "restored_sorted.csv", "restored_animal_sorted.csv", common_plants, network_type = "restored", output_txt_file_suffix="_community_details.txt")

    plt.title("Bipartite Pollination Graph - Restored Network with Communities", fontsize=16)
    plt.savefig("restored_graph_with_communities.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.show()