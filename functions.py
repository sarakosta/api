import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as sc
import community as co
import matplotlib.cm as cm
import graph_tool.all as gt


# extract row labels, column labels and matrix given an adjacency matrix in a csv file
def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0,
                     encoding='ISO-8859-1')  # First column is row labels
    # Extract row labels, column labels, and matrix
    return df.index.tolist(), df.columns.tolist(), df.values

# creates the newtork given the adjacency matrix
def create_network(file_path):
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

# draw bar chart for centrality measures of each species
def bar_chart(species, centrality_measure, name, common_names):
    plt.figure(figsize=(20, 10), dpi=100)
    bars = plt.bar(species, centrality_measure,
                   color='lightcoral', edgecolor='black')

    # color bars if they are in common
    for bar, label in zip(bars, species):
        if label in common_names:
            bar.set_color('blue')  # Highlighted color
        else:
            bar.set_color('red')  # Default color

    plt.xticks(rotation=90)  # Rotate x labels for readability

    ax = plt.gca()  # get current axis
    for tick in ax.get_xticklabels():
        if tick.get_text() in common_names:
            tick.set_color('blue')  # Highlighted label
        else:
            tick.set_color('red')   # Default label
    plt.xlabel("Species")
    plt.ylabel(f'{name}')
    plt.title(f'{name} Species')
    plt.tight_layout()  # Fix layout for better spacing
    plt.show()

# draw histogram for controlled and restored side by side
def histo_side_by_side(data_controlled, data_restored, data_name, kingdom_name):
    # create a unique array from the two separate data sets
    all_data = np.concatenate([data_controlled, data_restored])

    # extract the minimum and maximum values from the full data set
    min_val = int(np.floor(all_data.min()))
    max_val = int(np.ceil(all_data.max()))

    # define bins
    bins_data = np.arange(min_val - 0.5, max_val + 1.5, 1)

    # create and draw figure
    plt.figure(figsize=(8, 5))
    plt.hist(
        [data_controlled, data_restored],
        bins=bins_data,
        label=['Controlled', 'Restored'],
        align='left',
        edgecolor='black',
        alpha=0.7
    )
    plt.title(f'{data_name} for {kingdom_name}')
    plt.xlabel(f'{kingdom_name} {data_name}')
    plt.ylabel('Occurences')
    plt.legend()
    plt.grid(True)
    plt.show()


# define a bipartitite weighted Erdos-Renyi graph with same density
# as our graphs and weights sampled from the adjacency matrix
def create_erdos_renyi(file_path):
    rows, cols, adj_matrix = load_adjacency_matrix(file_path)

    # Set numbers
    num_plants = len(cols)
    num_pollinators = len(rows)
    interactions = np.count_nonzero(adj_matrix)
    # Probability of interaction
    p = interactions / (num_plants * num_pollinators)

    # Create bipartite graph
    G_er = nx.bipartite.random_graph(num_plants, num_pollinators, p)

    # Extract the weights from the adjacency matrix
    weights = adj_matrix[adj_matrix > 0]

    edges = list(G_er.edges())
    sampled_weights = np.random.choice(weights, size=len(edges), replace=True)

    # Assign weights as edge attributes
    for (edge, weight) in zip(edges, sampled_weights):
        G_er[edge[0]][edge[1]]['weight'] = weight

    return G_er


# -- CENTRALITY MEASURES --

# get the degree from the graph
def degree(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    # Get degrees of all nodes
    plant_degrees = [graph.degree(p) for p in plants]
    pollinator_degrees = [graph.degree(p) for p in pollinators]

    return plant_degrees, pollinator_degrees

def binomial_fit(degrees):
    degrees = np.array(degrees)
    sc.fit(sc.binom, degrees)

    fit_result = sc.fit(sc.binom, degrees, bounds={"n":[35, 40]})
    return fit_result

# get the weighted degree from the graph
def weighted_degree(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    # Get degrees of all nodes
    plant_w_degrees = [graph.degree(p, weight='weight') for p in plants]
    pollinator_w_degrees = [graph.degree(
        p, weight='weight') for p in pollinators]

    return plant_w_degrees, pollinator_w_degrees

# get the betweenness centrality from the graph
def betweenness_centrality(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    betweenness = nx.betweenness_centrality(graph, weight='weight')
    plant_bcw = {n: betweenness[n] for n in plants}
    pollinator_bcw = {n: betweenness[n] for n in pollinators}

    return plant_bcw, pollinator_bcw

# get the closeness centrality from the graph
def closeness_centrality(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    closeness = nx.closeness_centrality(graph, distance='weight')
    plant_ccw = {n: closeness[n] for n in plants}
    pollinator_ccw = {n: closeness[n] for n in pollinators}
    return plant_ccw, pollinator_ccw

# function that given the number of runs, the adjecency matrix, the common plant and
# pollinators and the type of the graph (controlled or restored) computes and draw in bar
# charts all the centrality measures whose functions were defined above
def centrality_measures(N_ER, file_path, common_plants, common_pollinators, graph_type):
    rows, cols, adj_matrix = load_adjacency_matrix(file_path)

    num_plants = len(cols)
    num_pollinators = len(rows)

    sum_bc_plants = np.zeros((num_plants))
    sum_bc_pollinators = np.zeros((num_pollinators))

    sum_cc_plants = np.zeros((num_plants))
    sum_cc_pollinators = np.zeros((num_pollinators))

    sum_wd_plants = np.zeros((num_plants))
    sum_wd_pollinators = np.zeros((num_pollinators))

    for n in range(N_ER):
        # create Erdos-Renyi
        erdos_renyi_graph = create_erdos_renyi(file_path)

        # betweenness centrality
        bc_plants, bc_pollinators = betweenness_centrality(erdos_renyi_graph)
        bc_plants_values = list(bc_plants.values())
        bc_pollinators_values = list(bc_pollinators.values())
        for i in range(num_plants):
            sum_bc_plants[i] += bc_plants_values[i]
        for i in range(num_pollinators):
            sum_bc_pollinators[i] += bc_pollinators_values[i]

        # closeness centrality
        cc_plants, cc_pollinators = closeness_centrality(erdos_renyi_graph)
        cc_plants_values = list(cc_plants.values())
        cc_pollinators_values = list(cc_pollinators.values())
        for i in range(num_plants):
            sum_cc_plants[i] += cc_plants_values[i]
        for i in range(num_pollinators):
            sum_cc_pollinators[i] += cc_pollinators_values[i]

        # weighted degree
        wd_plants, wd_pollinators = weighted_degree(erdos_renyi_graph)
        for i in range(num_plants):
            sum_wd_plants[i] += wd_plants[i]
        for i in range(num_pollinators):
            sum_wd_pollinators[i] += wd_pollinators[i]

    # species names
    restored_graph = create_network(file_path)
    bc_plants, bc_pollinators = betweenness_centrality(restored_graph)
    species_plant = list(bc_plants.keys())
    species_pollinators = list(bc_pollinators.keys())

    # plot average betweennes centrality
    mean_bc_plants = sum_bc_plants / N_ER
    mean_bc_plants_name = f'Mean Betweennes Centrality for plants over {N_ER} Erdos-Renyi graphs'
    bar_chart(species_plant, mean_bc_plants,
              mean_bc_plants_name, common_plants)
    mean_bc_pollinators = sum_bc_pollinators / N_ER
    mean_bc_pollinators_name = f'Mean Betweennes Centrality for pollinators over {N_ER} Erdos-Renyi graphs'
    bar_chart(species_pollinators, mean_bc_pollinators,
              mean_bc_pollinators_name, common_pollinators)

    # betweennes centrality distribution for our network
    bc_plant_restored, bc_pollinator_restored = betweenness_centrality(
        restored_graph)
    bc_plants_values_restored = list(bc_plant_restored.values())
    bc_pollinators_values_restored = list(bc_pollinator_restored.values())
    bc_plants_name_restored = f'Betweennes Centrality for plants for the {graph_type} network'
    bar_chart(species_plant, bc_plants_values_restored,
              bc_plants_name_restored, common_plants)
    bc_pollinators_name_restored = f'Betweennes Centrality for pollinators for the {graph_type} network'
    bar_chart(species_pollinators, bc_pollinators_values_restored,
              bc_pollinators_name_restored, common_pollinators)

    # plot average closeness centrality
    mean_cc_plants = sum_cc_plants / N_ER
    mean_cc_plants_name = f'Mean Closeness Centrality for plants over {N_ER} Erdos-Renyi graphs'
    bar_chart(species_plant, mean_cc_plants,
              mean_cc_plants_name, common_plants)
    mean_cc_pollinators = sum_cc_pollinators / N_ER
    mean_cc_pollinators_name = f'Mean Closeness Centrality for pollinators over {N_ER} Erdos-Renyi graphs'
    bar_chart(species_pollinators, mean_cc_pollinators,
              mean_cc_pollinators_name, common_pollinators)

    # closeness centrality distribution for our network
    cc_plant_restored, cc_pollinator_restored = closeness_centrality(
        restored_graph)
    cc_plants_values_restored = list(cc_plant_restored.values())
    cc_pollinators_values_restored = list(cc_pollinator_restored.values())
    cc_plants_name_restored = f'Closeness Centrality for plants for the {graph_type} network'
    bar_chart(species_plant, cc_plants_values_restored,
              cc_plants_name_restored, common_plants)
    cc_pollinators_name_restored = f'Closeness Centrality for pollinators for the {graph_type} network'
    bar_chart(species_pollinators, cc_pollinators_values_restored,
              cc_pollinators_name_restored, common_pollinators)

    # plot average weighted degree
    mean_wd_plants = sum_wd_plants / N_ER
    mean_wd_plants_name = f'Mean Weighted Degree for plants over {N_ER} Erdos-Renyi graphs'
    bar_chart(species_plant, mean_wd_plants,
              mean_wd_plants_name, common_plants)
    mean_wd_pollinators = sum_wd_pollinators / N_ER
    mean_wd_pollinators_name = f'Mean Weighted Degree for pollinators over {N_ER} Erdos-Renyi graphs'
    bar_chart(species_pollinators, mean_wd_pollinators,
              mean_wd_pollinators_name, common_pollinators)

    # weighted degree distribution for our network
    plant_wd_api, pollinator_wd_api = weighted_degree(restored_graph)
    wd_plants_name_api = f'Weighted Degree for plants for the {graph_type} network'
    bar_chart(species_plant, plant_wd_api, wd_plants_name_api, common_plants)
    wd_pollinators_name_api = f'Weighted Degree for pollinators for the {graph_type} network'
    bar_chart(species_pollinators, pollinator_wd_api,
              wd_pollinators_name_api, common_pollinators)

    # mannwhitney test for p-value for weighted degree
    statistic_mw_wd, p_value_mw_wd = sc.mannwhitneyu(
        plant_wd_api, mean_wd_plants, alternative='two-sided')
    print(
        f'Weighted Degrees Mann-Withney for {graph_type} network', statistic_mw_wd, p_value_mw_wd)

    statistic_ks_wd, p_value_ks_wd = sc.ks_2samp(plant_wd_api, mean_wd_plants)
    print(
        f'Weighted Degrees Kolmogorov-Smirnov for {graph_type} network', statistic_ks_wd, p_value_ks_wd)

    # mannwhitney test for p-value for betweennes centrality
    statistic_mw_bc, p_value_mw_bc = sc.mannwhitneyu(
        bc_plants_values_restored, mean_bc_plants, alternative='two-sided')
    print(
        f'Betweennees Centrality Mann-Withney for {graph_type} network', statistic_mw_bc, p_value_mw_bc)

    statistic_ks_bc, p_value_ks_bc = sc.ks_2samp(
        bc_plants_values_restored, mean_bc_plants)
    print(
        f'Betweennees Centrality Kolmogorov-Smirnov for {graph_type} network', statistic_ks_bc, p_value_ks_bc)

    # mannwhitney test for p-value for closeness centrality
    statistic_mw_cc, p_value_mw_cc = sc.mannwhitneyu(
        cc_plants_values_restored, mean_cc_plants, alternative='two-sided')
    print(
        f'Closenness Centrality Mann-Withney for {graph_type} network', statistic_mw_bc, p_value_mw_cc)

    statistic_ks_cc, p_value_ks_cc = sc.ks_2samp(
        cc_plants_values_restored, mean_cc_plants)
    print(
        f'Closeness Centrality Kolmogorov-Smirnov for {graph_type} network', statistic_ks_cc, p_value_ks_cc)

 # -- FUNCTIONS TO DO COSMETIC ON THE GRAPHS --

 # adjust the node side to be bigger with bigger flora abundance (plants) or animal numbers (pollinators)
def compute_node_sizes(G, plant_read, pollinator_read, min_size_plant=100, min_size_pollinator=100):
    plant_sizes = []
    pollinator_sizes = []
    node_sizes = {}

    for node in G.nodes:
        if G.nodes[node]["bipartite"] == 0:
            abundance_value = plant_read.loc[
                plant_read.iloc[:, 2] == node,
                plant_read.columns[7]
            ].values
            size = float(abundance_value[0]) if len(
                abundance_value) > 0 else 1.0
            scaled_size = max(size * 3000, min_size_plant)
            node_sizes[node] = scaled_size
            plant_sizes.append(scaled_size)
        else:
            n_visits = pollinator_read.loc[
                pollinator_read.iloc[:, 3] == node,
                pollinator_read.columns[6]
            ].values
            size = float(n_visits[0]) if len(n_visits) > 0 else 1.0
            scaled_size = max(size * 10, min_size_pollinator)
            node_sizes[node] = scaled_size
            pollinator_sizes.append(scaled_size)

    return node_sizes, plant_sizes, pollinator_sizes

# adjust the positioning of the nodes not to have superposition
def compute_vertical_positions(sizes, min_spacing=0.02, scale_factor=100):
    total_size = sum(sizes)
    normalized_sizes = [s / total_size for s in sizes]

    centers = []
    y = 0
    for h in normalized_sizes:
        y += h / 2
        centers.append(y)
        y += h / 2 + min_spacing

    mid_y = (centers[0] + centers[-1]) / 2
    centered_positions = [c - mid_y for c in centers]
    scaled_positions = [y * scale_factor for y in centered_positions]

    return scaled_positions


def nodes_and_edges(file_adj_matrix,
                    plant_file,
                    pollinator_file,
                    min_spacing,
                    min_size,
                    scale_factor):

    G = create_network(file_adj_matrix)
    plant_read = pd.read_csv(plant_file, encoding='ISO-8859-1')
    pollinator_read = pd.read_csv(pollinator_file, encoding='ISO-8859-1')

    plants = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    pollinators = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

    node_sizes, plant_sizes, pollinator_sizes = compute_node_sizes(
        G, plant_read, pollinator_read, min_size_plant=min_size, min_size_pollinator=min_size
    )

    plant_y = compute_vertical_positions(
        plant_sizes, min_spacing, scale_factor)
    pollinator_y = compute_vertical_positions(
        pollinator_sizes, min_spacing, scale_factor)

    plant_pos = {plant: (0, y) for plant, y in zip(plants, plant_y)}
    pollinator_pos = {pol: (1, y) for pol, y in zip(pollinators, pollinator_y)}
    pos = {**plant_pos, **pollinator_pos}

    # Adjust the thickness of the edges based on the interaction frequency.
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    max_weight = max(weights) if weights else 1
    normalized_weights = [0.5 + (w / max_weight) * 5 for w in weights]

    return G, pos, node_sizes, normalized_weights, plant_read, pollinator_read


# function that draws the network colouring the animal nodes differently depending on their order
# and the plants depending if they are in common or not in the two sites
def draw_network_order(
    file_adj_matrix,
    plant_file,
    pollinator_file,
    common_names,
    min_spacing=0.02,
    min_size=100,
    scale_factor=100
):
    G, pos, node_sizes, normalized_weights, plant_read, pollinator_read = nodes_and_edges(
        file_adj_matrix, plant_file, pollinator_file, min_spacing, min_size, scale_factor)

    # identify order of the animal species and color code them
    species_to_order = {
        species.strip(): order.strip()
        for species, order in zip(pollinator_read.iloc[:, 3], pollinator_read.iloc[:, 1])
    }

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

    # If the plants are in common between the controlled and restored site,
    # they will be blue, if not red.
    # The pollinators are colored based on the order of the species.
    node_colors = []
    for node in G.nodes:
        if G.nodes[node]['bipartite'] == 0:
            color = "blue" if node in common_names else "red"
        else:
            order = species_to_order.get(node.strip())
            color = order_color_map.get(order, default_color)
        node_colors.append(color)

    plt.figure(figsize=(20, 60))
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=[node_sizes[n] for n in G.nodes],
        node_color=node_colors,
        edge_color="black",
        width=normalized_weights
    )

    label_offset = 0.02
    for node, (x, y) in pos.items():
        ha = 'right' if G.nodes[node]['bipartite'] == 0 else 'left'
        offset = -label_offset if ha == 'right' else label_offset
        plt.text(x + offset, y, node, ha=ha, va='center', fontsize=30)

    legend_patches = [mpatches.Patch(color=c, label=o)
                      for o, c in order_color_map.items()]
    legend_patches.append(mpatches.Patch(
        color="blue", label="Highlighted Plants"))
    legend_patches.append(mpatches.Patch(color="red", label="Other Plants"))

    plt.legend(handles=legend_patches, loc='upper left',
               bbox_to_anchor=(0.05, 0.15), fontsize=30)
    plt.axis('off')


def draw_network_communities(
    file_adj_matrix,
    plant_file,
    pollinator_file,
    partition,
    num_communities,
    communities_grouped,
    network_type,
    min_spacing=0.02,
    min_size=100,
    scale_factor=100
):
    G, pos, node_sizes, normalized_weights, plant_read, pollinator_read = nodes_and_edges(
        file_adj_matrix, plant_file, pollinator_file, min_spacing, min_size, scale_factor)

    # Prepare node colors based on community
    community_colors = cm.get_cmap('tab20', num_communities)
    node_community_colors = [community_colors(
        partition[node]) for node in G.nodes()]

    # --- Plotting the main network graph ---
    plt.figure(figsize=(20, 60))
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
            plt.text(x - label_offset, y, node,
                     ha='right', va='center', fontsize=20)
        else:
            plt.text(x + label_offset, y, node,
                     ha='left', va='center', fontsize=20)

    community_patches = []
    for i in range(num_communities):
        community_patches.append(mpatches.Patch(
            color=community_colors(i), label=f'Community {i}'))

    legend_handles = community_patches
    plt.legend(handles=legend_handles, loc='upper left',
               bbox_to_anchor=(0.05, 0.15), fontsize=30)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # plt.show()

    # --- DRAWING INDIVIDUAL GRAPHS AND CREATING ADJACENCY MATRICES FOR EACH COMMUNITY ---
    print("\n--- Processing Individual Community Graphs and Adjacency Matrices ---")
    for comm_id, nodes_in_comm in communities_grouped.items():
        # Filter nodes into plants and pollinators within this community
        plants_in_comm = [
            n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 0]
        pollinators_in_comm = [
            n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 1]

        # Create a subgraph containing only nodes from the current community
        subgraph = G.subgraph(nodes_in_comm)

        # Skip if the community is empty or has no relevant edges for an adjacency matrix
        if not plants_in_comm or not pollinators_in_comm or subgraph.number_of_edges() == 0:
            if not nodes_in_comm:
                print(f"Community {comm_id} is empty. Skipping plot and CSV.")
            elif subgraph.number_of_edges() == 0:
                print(
                    f"Community {comm_id} has nodes but no internal edges. Skipping plot and CSV.")
            elif not plants_in_comm:
                print(f"Community {comm_id} has no plants. Skipping CSV.")
            elif not pollinators_in_comm:
                print(f"Community {comm_id} has no pollinators. Skipping CSV.")
            continue  # Skip to next community if no valid bipartite interactions

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
            # Get weight, default to 0 if not present
            weight = data.get('weight', 0)
            # Determine which is pollinator and which is plant
            # u is pollinator, v is plant
            if G.nodes[u]['bipartite'] == 1 and G.nodes[v]['bipartite'] == 0:
                community_adj_df.loc[u, v] = weight
            # u is plant, v is pollinator
            elif G.nodes[u]['bipartite'] == 0 and G.nodes[v]['bipartite'] == 1:
                # Always pollinator (row) to plant (column)
                community_adj_df.loc[v, u] = weight

        # Save the community's adjacency matrix to a CSV file
        csv_filename = f"{network_type}_community_{comm_id}_adjacency_matrix.csv"
        community_adj_df.to_csv(csv_filename, encoding='utf-8')


# -- FUNCTIONS TO COMPUTE EVENNESS --

def shannon_entropy(prob):
    # Add a small value to avoid log(0)
    entropy = -np.sum(prob * np.log2(prob + 1e-9))
    return entropy


def evenness(file_path):
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_path)
    prob = adj_matrix/np.sum(adj_matrix)
    evenness = shannon_entropy(prob) / np.log2(np.count_nonzero(adj_matrix))
    return evenness

# -- COMMUNITY DETECTION --

# define a graph_tool graph from the NetworkX graph


def graph_gt(G):
    g_gt = gt.Graph(directed=False)

    # Create vertex property maps for node name, bipartite partition, and weight
    gt_name_prop = g_gt.new_vertex_property("string")
    gt_bipartite_prop = g_gt.new_vertex_property("int")
    gt_weight_prop = g_gt.new_edge_property("double")  # For edge weights

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
        # Use 1.0 as default weight if not present
        gt_weight_prop[e] = data.get('weight', 1.0)

    g_gt.vp.name = gt_name_prop
    g_gt.vp.bipartite = gt_bipartite_prop
    g_gt.ep.weight = gt_weight_prop

    return g_gt


def Louvain_method(G):
    partition = co.best_partition(G, weight='weight')
    num_communities = max(partition.values()) + 1
    modularity = co.modularity(partition, G, weight='weight')
    return partition, num_communities, modularity

# performs luovain method for communities N_louvain times, selects the run that maximizes
# modularity and prints in a txt file the communities and modularity, ...


def print_Louvain_communities(G, N_louvain, network_type,
                              output_txt_file_suffix="_Louvain_community_details.txt"):

    partition, num_communities, modularity = Louvain_method(G)
    for n in range(N_louvain):
        partition_new, num_communities_new, modularity_new = Louvain_method(G)
        if modularity < modularity_new:
            partition = partition_new
            num_communities = num_communities_new
            modularity = modularity_new

    # --- Print Community Details to console AND file ---
    communities_grouped = {i: [] for i in range(num_communities)}
    for node, comm_id in partition.items():
        communities_grouped[comm_id].append(node)

    # Construct full filename
    full_output_txt_filename = f"{network_type}{output_txt_file_suffix}"
    with open(full_output_txt_filename, 'w', encoding='utf-8') as f:
        # Added network_type for clarity in output text
        f.write(
            f"\nNetwork loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. ({network_type} network)" + '\n')
        f.write("\n--- Performing Community Detection ---" + '\n')
        f.write(f"Detected {num_communities} communities." + '\n')
        f.write(f"Modularity of the partition: {modularity:.4f}" + '\n')
        f.write("\n--- Community Details ---" + '\n')

        for comm_id, nodes_in_comm in communities_grouped.items():
            plants_in_comm = [
                n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 0]
            pollinators_in_comm = [
                n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 1]
            f.write(f"Community {comm_id}:" + '\n')
            f.write(
                f"  Plants ({len(plants_in_comm)}): {', '.join(plants_in_comm) if plants_in_comm else 'None'}" + '\n')
            f.write(
                f"  Pollinators ({len(pollinators_in_comm)}): {', '.join(pollinators_in_comm) if pollinators_in_comm else 'None'}" + '\n')
            f.write("-" * 30 + '\n')

    return partition, num_communities, modularity, communities_grouped


def print_biSBM_communities(G, N_biSBM, network_type,
                            output_txt_file_suffix="_biSBM_community_details.txt"):

    # get a graph tool graph from the networkx one
    g_gt = graph_gt(G)

    bisbm_state = gt.minimize_blockmodel_dl(
        g_gt, state_args={'bipartite_nodes': g_gt.vp.bipartite})
    optimal_bisbm_dl = bisbm_state.entropy()

    for n in range(N_biSBM):
        bisbm_state_new = gt.minimize_blockmodel_dl(
            g_gt, state_args={'bipartite_nodes': g_gt.vp.bipartite})
        optimal_bisbm_dl_new = bisbm_state_new.entropy()
        if optimal_bisbm_dl > optimal_bisbm_dl_new:
            bisbm_state = bisbm_state_new
            optimal_bisbm_dl = optimal_bisbm_dl_new

    # Extract the partition from the optimal bisbm_state
    # The block assignments are stored in the 'b' property map of the state object
    # We need to map graph-tool vertex IDs back to NetworkX node names
    partition = {}
    # gt_node_map_reverse = {v: k for k, v in g_gt.vp.name.items()} # Create reverse map

    for v in g_gt.vertices():
        node_name = g_gt.vp.name[v]
        community_id = bisbm_state.get_blocks()[v]
        # Ensure integer for consistency
        partition[node_name] = int(community_id)

    # Determine the number of communities
    # In graph-tool SBM, get_B() gives the number of blocks
    num_communities = bisbm_state.get_B()

    # --- Print Community Details to file ---
    communities_grouped = {i: [] for i in range(num_communities)}
    for node, comm_id in partition.items():
        communities_grouped[comm_id].append(node)

    # Construct full filename
    full_output_txt_filename = f"{network_type}{output_txt_file_suffix}"
    with open(full_output_txt_filename, 'w', encoding='utf-8') as f:
        f.write(
            f"\nNetwork loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. ({network_type} network)" + '\n')
        f.write(
            "\n--- Performing Bi-directional Stochastic Block Model Community Detection ---" + '\n')
        f.write(f"Detected {num_communities} communities." + '\n')
        f.write(
            f"Optimal Description Length (Entropy): {optimal_bisbm_dl:.4f}" + '\n')
        f.write("\n--- Community Details ---" + '\n')

        for comm_id, nodes_in_comm in communities_grouped.items():
            plants_in_comm = [
                n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 0]
            pollinators_in_comm = [
                n for n in nodes_in_comm if G.nodes[n]['bipartite'] == 1]
            f.write(f"Community {comm_id}:" + '\n')
            f.write(
                f"  Plants ({len(plants_in_comm)}): {', '.join(plants_in_comm) if plants_in_comm else 'None'}" + '\n')
            f.write(
                f"  Pollinators ({len(pollinators_in_comm)}): {', '.join(pollinators_in_comm) if pollinators_in_comm else 'None'}" + '\n')
            f.write("-" * 30 + '\n')

    modularity = co.modularity(partition, G, weight='weight')

    return partition, num_communities, optimal_bisbm_dl, communities_grouped, modularity, bisbm_state

def projections_old(G):
    # Perform the unipartite projection
    plant_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    pollinator_nodes = {n for n, d in G.nodes(
        data=True) if d['bipartite'] == 1}
    
    # Project the graph onto the first family of nodes
    G_plants = nx.bipartite.weighted_projected_graph(G, plant_nodes, )
    
    # Project the graph onto the second family of nodes
    G_pollinators = nx.bipartite.weighted_projected_graph(G, pollinator_nodes)
    
    return G_plants, G_pollinators

def projections(G):
    adj_matrix = nx.adjacency_matrix(G)
    adj_matrix = adj_matrix.toarray()
    
    plants_proj_adj_matrix = adj_matrix.T @ adj_matrix
    num_nodes_plants = plants_proj_adj_matrix.shape[0]
    G_plants = nx.Graph()
    G_plants.add_nodes_from(range(num_nodes_plants))
    # Add edges based on the adjacency matrix
    for i in range(num_nodes_plants):
        for j in range(num_nodes_plants):
            weight = plants_proj_adj_matrix[i, j]
            if weight != 0:
                G_plants.add_edge(i, j, weight=weight)
    
    pollinators_proj_adj_matrix = adj_matrix @ adj_matrix.T
    num_nodes_pollinators = pollinators_proj_adj_matrix.shape[0]
    G_pollinators = nx.Graph()
    G_pollinators.add_nodes_from(range(num_nodes_pollinators))
    # Add edges based on the adjacency matrix
    for i in range(num_nodes_pollinators):
        for j in range(num_nodes_pollinators):
            weight = pollinators_proj_adj_matrix[i, j]
            if weight != 0:
                G_pollinators.add_edge(i, j, weight=weight)
    
    return G_plants, G_pollinators
    

def print_SBM_communities(G, N_SBM, network_type,
                          output_txt_file_plant_suffix="_plant_projection_SBM_community_details.txt",
                          output_txt_file_pollinator_suffix="_pollinator_projection_SBM_community_details.txt"):

    G_plants, G_pollinators = projections_old(G)

    # convert to graph_tool graph
    G_gt_plants = graph_gt(G_plants)
    G_gt_pollinators = graph_gt(G_pollinators)

    # minimize description length
    state_plants_unipartite = gt.minimize_blockmodel_dl(
        G_gt_plants, state_args={'verbose': True})
    state_pollinators_unipartite = gt.minimize_blockmodel_dl(
        G_gt_pollinators, state_args={'verbose': True})

    # plants
    optimal_sbm_dl_plants = state_plants_unipartite.entropy()
    for n in range(N_SBM):
        state_plants_unipartite_new = gt.minimize_blockmodel_dl(
            G_gt_plants, state_args={'verbose': True})
        optimal_sbm_dl_plants_new = state_plants_unipartite_new.entropy()
        if optimal_sbm_dl_plants > optimal_sbm_dl_plants_new:
            state_plants_unipartite = state_plants_unipartite_new
            optimal_sbm_dl_plants = optimal_sbm_dl_plants_new

    # pollinators
    optimal_sbm_dl_pollinators = state_pollinators_unipartite.entropy()
    for n in range(N_SBM):
        state_pollinators_unipartite_new = gt.minimize_blockmodel_dl(
            G_gt_pollinators, state_args={'verbose': True})
        optimal_sbm_dl_pollinators_new = state_pollinators_unipartite_new.entropy()
        if optimal_sbm_dl_pollinators > optimal_sbm_dl_pollinators_new:
            state_pollinators_unipartite = state_pollinators_unipartite_new
            optimal_sbm_dl_pollinators = optimal_sbm_dl_pollinators_new

    # A^T*A
    partition_plants = {}
    # gt_node_map_reverse = {v: k for k, v in g_gt.vp.name.items()} # Create reverse map

    for v in G_gt_plants.vertices():
        node_name = G_gt_plants.vp.name[v]
        community_id_plants = state_plants_unipartite.get_blocks()[v]
        # Ensure integer for consistency
        partition_plants[node_name] = int(community_id_plants)

    # Determine the number of communities
    # In graph-tool SBM, get_B() gives the number of blocks
    num_communities_plants = state_plants_unipartite.get_B()

    # --- Print Community Details to file ---
    communities_grouped_plants = {i: [] for i in range(num_communities_plants)}
    for node, comm_id in partition_plants.items():
        communities_grouped_plants[comm_id].append(node)

    # Construct full filename
    full_output_txt_filename = f"{network_type}{output_txt_file_plant_suffix}"
    with open(full_output_txt_filename, 'w', encoding='utf-8') as f:
        f.write(
            f"\nNetwork loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. ({network_type} network)" + '\n')
        f.write(
            "\n--- Performing Stochastic Block Model Community Detection for A^T*A ---" + '\n')
        f.write(f"Detected {num_communities_plants} communities." + '\n')
        f.write(
            f"Optimal Description Length (Entropy): {optimal_sbm_dl_plants:.4f}" + '\n')
        f.write("\n--- Community Details ---" + '\n')

        for comm_id, nodes_in_comm in communities_grouped_plants.items():
            f.write(f"Community {comm_id}:" + '\n')
            f.write(f"  Plants ({len(nodes_in_comm)}: {', '.join(nodes_in_comm) if nodes_in_comm else 'None'}" + '\n')
            f.write("-" * 30 + '\n')

    modularity_plants = co.modularity(partition_plants, G_plants, weight='weight')

    # A*A^T
    partition_pollinators = {}
    # gt_node_map_reverse = {v: k for k, v in g_gt.vp.name.items()} # Create reverse map

    for v in G_gt_pollinators.vertices():
        node_name = G_gt_pollinators.vp.name[v]
        community_id_pollinators = state_pollinators_unipartite.get_blocks()[v]
        partition_pollinators[node_name] = int(
            community_id_pollinators)  # Ensure integer for consistency

    # Determine the number of communities
    # In graph-tool SBM, get_B() gives the number of blocks
    num_communities_pollinators = state_pollinators_unipartite.get_B()

    # --- Print Community Details to file ---
    communities_grouped_pollinators = {i: []
                                       for i in range(num_communities_pollinators)}
    for node, comm_id in partition_pollinators.items():
        communities_grouped_pollinators[comm_id].append(node)

    # Construct full filename
    full_output_txt_filename = f"{network_type}{output_txt_file_pollinator_suffix}"
    with open(full_output_txt_filename, 'w', encoding='utf-8') as f:
        f.write(
            f"\nNetwork loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. ({network_type} network)" + '\n')
        f.write(
            "\n--- Performing Stochastic Block Model Community Detection for A^T*A ---" + '\n')
        f.write(f"Detected {num_communities_pollinators} communities." + '\n')
        f.write(
            f"Optimal Description Length (Entropy): {optimal_sbm_dl_pollinators:.4f}" + '\n')
        f.write("\n--- Community Details ---" + '\n')

        for comm_id, nodes_in_comm in communities_grouped_pollinators.items():
            f.write(f"Community {comm_id}:" + '\n')
            f.write(f"  Plants ({len(nodes_in_comm)}: {', '.join(nodes_in_comm) if nodes_in_comm else 'None'}" + '\n')
            f.write("-" * 30 + '\n')

    modularity_pollinators = co.modularity(
        partition_pollinators, G_pollinators, weight='weight')

    return partition_plants, num_communities_plants, optimal_sbm_dl_plants, communities_grouped_plants, modularity_plants, state_plants_unipartite, partition_pollinators, num_communities_plants, optimal_sbm_dl_pollinators, communities_grouped_pollinators, modularity_pollinators, state_pollinators_unipartite
