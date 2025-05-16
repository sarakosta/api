import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sc

# extract row labels, column labels and matrix given an adjacency matrix in a csv file
def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')  # First column is row labels
    return df.index.tolist(), df.columns.tolist(), df.values  # Extract row labels, column labels, and matrix

# nel futuro mettere df.values come primo output e cambiare tutto il codice di conseguenza

def weight_distribution(file_path):
    plants, pollinators, adj_matrix = load_adjacency_matrix(file_path)
    weights_real = adj_matrix[adj_matrix > 0]
    return weights_real 

# creates the newtork given the adjacency matrix
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

def erdos_renyi_native(file_path, weights_real):
    rows, cols, adj_matrix = load_adjacency_matrix(file_path)

    # Set numbers
    num_plants = len(cols)
    num_pollinators = len(rows)
    interactions = np.count_nonzero(adj_matrix)
    p = interactions / (num_plants * num_pollinators)  # Probability of interaction

    # Create bipartite graph
    G_er = nx.bipartite.random_graph(num_plants, num_pollinators, p)
    
    edges = list(G_er.edges())
    sampled_weights = np.random.choice(weights_real, size=len(edges), replace=True)

    # Assign weights as edge attributes
    for (edge, weight) in zip(edges, sampled_weights):
        G_er[edge[0]][edge[1]]['weight'] = weight
        
    return G_er  

# plot the degree of a graph
def degree(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    # Get degrees of all nodes
    plant_degrees = [graph.degree(p) for p in plants]
    pollinator_degrees = [graph.degree(p) for p in pollinators]
    
    return plant_degrees, pollinator_degrees

# draw an histogram
def histo(plant_degrees, pollinator_degrees):
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(plant_degrees, bins=range(0, max(plant_degrees)+2), align='left', edgecolor='black')
    plt.title('Degree Distribution for Plants')
    plt.xlabel('Plant Degree (number of interactions)')
    plt.ylabel('Number of plant nodes')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.hist(pollinator_degrees, bins=range(0, max(pollinator_degrees)+2), align='left', edgecolor='black')
    plt.title('Degree Distribution for Pollinators')
    plt.xlabel('Pollinator Degree (number of interactions)')
    plt.ylabel('Number of pollinator nodes')
    plt.grid(True)
    plt.show()
    
# compute some centrality measures
def weighted_degree(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}

    # Get degrees of all nodes
    plant_w_degrees = [graph.degree(p, weight='weight') for p in plants]
    pollinator_w_degrees = [graph.degree(p, weight='weight') for p in pollinators]
    
    return plant_w_degrees, pollinator_w_degrees

# draw bar chart for centrality measures of each species
def bar_chart(species, centrality_measure, name, common_names):
    plt.figure(figsize=(20, 10), dpi = 100)
    bars = plt.bar(species, centrality_measure, color='lightcoral', edgecolor='black')
    
    # color bars if they are in common
    for bar, label in zip(bars, species):
        if label in common_names:
            bar.set_color('blue')  # Highlighted color
        else:
            bar.set_color('red')  # Default color
    
    plt.xticks(rotation=90)  # Rotate x labels for readability
    
    ax = plt.gca() # get current axis
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

# centrality measures
def betweenness_centrality(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}
    
    betweenness = nx.betweenness_centrality(graph, weight = 'weight')
    plant_bcw = {n: betweenness[n] for n in plants}
    pollinator_bcw = {n: betweenness[n] for n in pollinators}
    
    return plant_bcw, pollinator_bcw

def closeness_centrality(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}
    
    closeness = nx.closeness_centrality(graph, distance = 'weight')
    plant_ccw = {n: closeness[n] for n in plants}
    pollinator_ccw = {n: closeness[n] for n in pollinators}
    return plant_ccw, pollinator_ccw

# define an array of N_ER Erdos Renyi networks

N_ER = 100

def centrality_measures(N_ER, file_path, common_plants, common_pollinators):
    rows, cols, adj_matrix = load_adjacency_matrix(file_path)
    weights = weight_distribution(file_path)
    
    num_plants = len(cols)
    num_pollinators = len(rows)
    
    sum_bc_plants = np.zeros((num_plants))
    sum_bc_pollinators = np.zeros((num_pollinators))
    
    sum_cc_plants = np.zeros((num_plants))
    sum_cc_pollinators = np.zeros((num_pollinators))
    
    sum_wd_plants = np.zeros((num_plants))
    sum_wd_pollinators = np.zeros((num_pollinators))
    
    for n in range(N_ER):
        # create Erod Renyi
        erdos_renyi_graph = erdos_renyi_native(file_path, weights)
        
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
        #wd_plants_values = list(wd_plants.values())
        #wd_pollinators_values = list(wd_pollinators.values())
        for i in range(num_plants):
            sum_wd_plants[i] += wd_plants[i]
        for i in range(num_pollinators):
            sum_wd_pollinators[i] += wd_pollinators[i]
        
    # species names    
    restored_graph = network(file_path)
    bc_plants, bc_pollinators = betweenness_centrality(restored_graph)
    species_plant = list(bc_plants.keys())
    species_pollinators = list(bc_pollinators.keys())
    
    # plot average betweennes centrality
    mean_bc_plants = sum_bc_plants / N_ER
    mean_bc_plants_name = "Mean BC for plants over ER"
    bar_chart(species_plant, mean_bc_plants, mean_bc_plants_name, common_plants)
    mean_bc_pollinators = sum_bc_pollinators / N_ER
    mean_bc_pollinators_name = "Mean BC for pollinators over ER"
    bar_chart(species_pollinators, mean_bc_pollinators, mean_bc_pollinators_name, common_pollinators)
    
    # betweennes centrality distribution for our network
    bc_plant_restored, bc_pollinator_restored = betweenness_centrality(restored_graph)
    bc_plants_values_restored = list(bc_plant_restored.values())
    bc_pollinators_values_restored = list(bc_pollinator_restored.values())
    bc_plants_name_restored = "BC for plants for our network"
    bar_chart(species_plant, bc_plants_values_restored, bc_plants_name_restored, common_plants)
    bc_pollinators_name_restored = "BC for pollinators for our network"
    bar_chart(species_pollinators, bc_pollinators_values_restored, bc_pollinators_name_restored, common_pollinators)
    
    # plot average closeness centrality
    mean_cc_plants = sum_cc_plants / N_ER
    mean_cc_plants_name = "Mean CC for plants over ER"
    bar_chart(species_plant, mean_cc_plants, mean_cc_plants_name, common_plants)
    mean_cc_pollinators = sum_cc_pollinators / N_ER
    mean_cc_pollinators_name = "Mean CC for pollinators over ER"
    bar_chart(species_pollinators, mean_cc_pollinators, mean_cc_pollinators_name, common_pollinators)
    
    # closeness centrality distribution for our network
    cc_plant_restored, cc_pollinator_restored = closeness_centrality(restored_graph)
    cc_plants_values_restored = list(cc_plant_restored.values())
    cc_pollinators_values_restored = list(cc_pollinator_restored.values())
    cc_plants_name_restored = "CC for plants for our network"
    bar_chart(species_plant, cc_plants_values_restored, cc_plants_name_restored, common_plants)
    cc_pollinators_name_restored = "CC for pollinators for our network"
    bar_chart(species_pollinators, cc_pollinators_values_restored, cc_pollinators_name_restored, common_pollinators)
    
    # plot average weighted degree
    mean_wd_plants = sum_wd_plants / N_ER
    mean_wd_plants_name = "Mean WD for plants over ER"
    bar_chart(species_plant, mean_wd_plants, mean_wd_plants_name, common_plants)
    mean_wd_pollinators = sum_wd_pollinators / N_ER
    mean_wd_pollinators_name = "Mean WD for pollinators over ER"
    bar_chart(species_pollinators, mean_wd_pollinators, mean_wd_pollinators_name, common_pollinators)
    
    # weighted degree distribution for our network
    plant_wd_api, pollinator_wd_api = weighted_degree(restored_graph)
    wd_plants_name_api = "WD for plants for our network"
    bar_chart(species_plant, plant_wd_api, wd_plants_name_api, common_plants)
    wd_pollinators_name_api = "WD for pollinators for our network"
    bar_chart(species_pollinators, pollinator_wd_api, wd_pollinators_name_api, common_pollinators)
    
    # mannwhitney test for p-value for weighted degree
    statistic_mw_wd, p_value_mw_wd = sc.mannwhitneyu(plant_wd_api, mean_wd_plants, alternative='two-sided')
    print("weighted degrees Mann-Withney", statistic_mw_wd, p_value_mw_wd)
    
    statistic_ks_wd, p_value_ks_wd = sc.ks_2samp(plant_wd_api, mean_wd_plants)
    print("weighted degrees Kolmogorov-Smirnov", statistic_ks_wd, p_value_ks_wd)
    
    # mannwhitney test for p-value for betweennes centrality
    statistic_mw_bc, p_value_mw_bc = sc.mannwhitneyu(bc_plants_values_restored, mean_bc_plants, alternative='two-sided')
    print("betweennes centrality Mann-Withney", statistic_mw_bc, p_value_mw_bc)
    
    statistic_ks_bc, p_value_ks_bc = sc.ks_2samp(bc_plants_values_restored, mean_bc_plants)
    print("betweennes centrality Kolmogorov-Smirnov", statistic_ks_bc, p_value_ks_bc)
    
    # mannwhitney test for p-value for closeness centrality
    statistic_mw_cc, p_value_mw_cc = sc.mannwhitneyu(cc_plants_values_restored, mean_cc_plants, alternative='two-sided')
    print("closeness centrality Mann-Withney", statistic_mw_bc, p_value_mw_cc)
    
    statistic_ks_cc, p_value_ks_cc = sc.ks_2samp(cc_plants_values_restored, mean_cc_plants)
    print("closeness centrality Kolmogorov-Smirnov",statistic_ks_cc, p_value_ks_cc)
    
pollinators_c, plants_c, adj_matrix_c = load_adjacency_matrix("gcontrolled.csv")
pollinators_r, plants_r, adj_matrix_r = load_adjacency_matrix("grestored.csv")

common_pollinators = set(pollinators_c) & set(pollinators_r)
common_plants = set(plants_c) & set(plants_r)

centrality_measures(N_ER, "grestored.csv", common_plants, common_pollinators)
centrality_measures(N_ER, "gcontrolled.csv", common_plants, common_pollinators)

