import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# extract row labels, column labels and matrix given an adjacency matrix in a csv file
def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')  # First column is row labels
    return df.index.tolist(), df.columns.tolist(), df.values  # Extract row labels, column labels, and matrix

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

# define an Erdos Renyi network with same probability of interaction of the original graph
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
def bar_chart(species, centrality_measure, name):
    # Extract node names and values
    # nodes = list(centrality_measure.keys())
    # values = list(centrality_measure.values())
    # Plot
    plt.figure(figsize=(20, 10), dpi = 100)
    plt.bar(species, centrality_measure, color='lightcoral', edgecolor='black')
    plt.xticks(rotation=90)  # Rotate x labels for readability
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

def eigenvector_centrality(graph):
    plants = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    pollinators = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}
    
    eigenvector_centrality = nx.eigenvector_centrality(graph, weight='weight', max_iter=1000)
    plant_ecw = {n: eigenvector_centrality[n] for n in plants}
    pollinator_ecw = {n: eigenvector_centrality[n] for n in pollinators}
    return plant_ecw, pollinator_ecw

# plant_w_degree, pollinator_w_degree = weighted_degree(restored_graph)    
# print(np.mean(plant_w_degree))
# print(np.mean(pollinator_w_degree))

# define an array of N_ER Erdos Renyi networks

N_ER = 10
rows, cols, adj_matrix = load_adjacency_matrix("grestored.csv")
num_plants = len(cols)
num_pollinators = len(rows)

sum_bc_plants = np.zeros((num_plants))
sum_bc_pollinators = np.zeros((num_pollinators))

sum_cc_plants = np.zeros((num_plants))
sum_cc_pollinators = np.zeros((num_pollinators))

sum_ec_plants = np.zeros((num_plants))
sum_ec_pollinators = np.zeros((num_pollinators))

for n in range(N_ER):
    # create Erod Renyi
    erdos_renyi_graph = erdos_renyi("grestored.csv")
    
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
        
    # eigenvector centrality
    ec_plants, ec_pollinators = eigenvector_centrality(erdos_renyi_graph)
    ec_plants_values = list(ec_plants.values())
    ec_pollinators_values = list(ec_pollinators.values())
    for i in range(num_plants):
        sum_ec_plants[i] += ec_plants_values[i]
    for i in range(num_pollinators):
        sum_ec_pollinators[i] += ec_pollinators_values[i]
    
# species names    
restored_graph = network("grestored.csv")
bc_plants, bc_pollinators = betweenness_centrality(restored_graph)
species_plant = list(bc_plants.keys())
species_pollinators = list(bc_pollinators.keys())

# plot average betweennes centrality
mean_bc_plants = sum_bc_plants / N_ER
mean_bc_plants_name = "Mean BC for plants over ER"
bar_chart(species_plant, mean_bc_plants, mean_bc_plants_name)
mean_bc_pollinators = sum_bc_pollinators / N_ER
mean_bc_pollinators_name = "Mean BC for pollinators over ER"
bar_chart(species_pollinators, mean_bc_pollinators, mean_bc_pollinators_name)

# plot average closeness centrality
mean_cc_plants = sum_cc_plants / N_ER
mean_cc_plants_name = "Mean CC for plants over ER"
bar_chart(species_plant, mean_cc_plants, mean_cc_plants_name)
mean_cc_pollinators = sum_cc_pollinators / N_ER
mean_cc_pollinators_name = "Mean CC for pollinators over ER"
bar_chart(species_pollinators, mean_cc_pollinators, mean_cc_pollinators_name)

# plot average eigenvector centrality
mean_ec_plants = sum_ec_plants / N_ER
mean_ec_plants_name = "Mean EC for plants over ER"
bar_chart(species_plant, mean_ec_plants, mean_ec_plants_name)
mean_ec_pollinators = sum_ec_pollinators / N_ER
mean_ec_pollinators_name = "Mean EC for pollinators over ER"
bar_chart(species_pollinators, mean_ec_pollinators, mean_ec_pollinators_name)

#ec_plants, ec_pollinators = eigenvector_centrality(restored_graph)
#ec_plants_values = list(ec_plants.values())
#ec_pollinators_values = list(ec_pollinators.values())
#ec_plants_name = "Eigenvector Centrality for Plant"
#ec_pollinators_name = "Betweenness Centrality for Pollinators"

#bar_chart(species_plant, ec_plants_values, ec_plants_name)
#bar_chart(species_pollinators ,ec_pollinators_values, ec_pollinators_name)

erdos_renyi_graph = erdos_renyi("grestored.csv")
plant_degree_er, pollinator_degree_er = degree(erdos_renyi_graph)

#restored_graph = network("grestored.csv")
#plant_degree_api, pollinator_degree_api = degree(restored_graph)

histo(plant_degree_er, pollinator_degree_er) 
#histo(plant_degree_api, pollinator_degree_api)

#between centrality
#bc_plants, bc_animals = betweenness_centrality(restored_graph)
#bc_plants_name = "Betweenness Centrality for Plant"
#bc_animals_name = "Betweenness Centrality for Plant"

#bar_chart(bc_plants, bc_plants_name)
#bar_chart(bc_animals, bc_animals_name)

