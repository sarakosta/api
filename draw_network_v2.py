#import numpy as np
import pandas as pd
#import networkx as nx
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import functions as f

#evennesses = []
# controlled_families = ["gcontrolled_coleoptera.csv", "gcontrolled_diptera.csv", "gcontrolled_hymenoptera.csv", "gcontrolled_lepidoptera.csv", "gcontrolled_squamata.csv"]
#for i in controlled_families:
#    append(evennesses, eveness(i))
#print(sum(evennesses)/len(evennesses))

pollinators_c, plants_c, adj_matrix_c = f.load_adjacency_matrix("data/gcontrolled.csv")
pollinators_r, plants_r, adj_matrix_r = f.load_adjacency_matrix("data/grestored.csv")

common_pollinators = set(pollinators_c) & set(pollinators_r)
common_plants = set(plants_c) & set(plants_r)

#evenness_c = list(map(evenness, controlled_families))
#print(evenness_c)
#print(sum(evenness_c)/len(controlled_families))

evenness_r = f.evenness("data/grestored.csv")
evenness_c = f.evenness("data/gcontrolled.csv")
print("controlled evenness:", evenness_c)
print("restored evenness:", evenness_r)

f.draw_network_order("data/gcontrolled.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", common_plants, min_spacing=0.02, min_size=50, scale_factor=150)
#plt.title("Bipartite Pollination Graph Controlled", fontsize = 100)    
# plt.savefig("controlled_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_order("data/grestored.csv", "data/restored_plants.csv", "data/restored_pollinators.csv" ,common_plants, min_spacing=0.02, min_size=70, scale_factor=150)
#plt.title("Bipartite Pollination Graph Restored", fontsize = 50)    
# plt.savefig("restored_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

plant_info_c = pd.read_csv("data/controlled_plants.csv", encoding='ISO-8859-1')
pollinator_info_c = pd.read_csv("data/controlled_pollinators.csv", encoding='ISO-8859-1')

plant_linkage_c = plant_info_c.iloc[:, 8]
plant_linkage_c = pd.to_numeric(plant_linkage_c, errors='coerce').dropna()

pollinator_linkage_c = pollinator_info_c.iloc[:, 8]
pollinator_linkage_c = pd.to_numeric(pollinator_linkage_c, errors='coerce').dropna()

plant_info_r = pd.read_csv("data/restored_plants.csv", encoding='ISO-8859-1')
pollinator_info_r = pd.read_csv("data/restored_pollinators.csv", encoding='ISO-8859-1')

plant_linkage_r = plant_info_r.iloc[:, 8]
plant_linkage_r = pd.to_numeric(plant_linkage_r, errors='coerce').dropna()

pollinator_linkage_r = pollinator_info_r.iloc[:, 8]
pollinator_linkage_r = pd.to_numeric(pollinator_linkage_r, errors='coerce').dropna()

linkage_name = "Linkage" 
plant_kingdom_name = "Plants"
f.histo_side_by_side(plant_linkage_c, plant_linkage_r, linkage_name, plant_kingdom_name)

G_c = f.create_network("data/gcontrolled.csv")
partition_l_c, num_communities_l_c, modularity_c, communities_grouped_l_c = f.print_Louvain_communities(G_c, N_louvain=1000, network_type="controlled")
partition_b_c, num_communities_b_c, optimal_bisbm_dl_c, communities_grouped_b_c, modularity_b_c = f.print_biSBM_communities(G_c, N_biSBM=1000, network_type="controlled")
G_r = f.create_network("data/grestored.csv")
partition_l_r, num_communities_l_r, modularity_r, communities_grouped_l_r = f.print_Louvain_communities(G_r, N_louvain=1000, network_type="restored")
partition_b_r, num_communities_b_r, optimal_bisbm_dl_r, communities_grouped_b_r, modularity_b_r = f.print_biSBM_communities(G_r, N_biSBM=1000, network_type="restored")

f.draw_network_communities(
    "data/gcontrolled.csv",
    "data/controlled_plants.csv",
    "data/controlled_pollinators.csv",
    partition_l_c,
    num_communities_l_c,
    communities_grouped_l_c,
    network_type="controlled",
    min_spacing=0.02,
    min_size=100,
    scale_factor=100)

f.draw_network_communities(
    "data/gcontrolled.csv",
    "data/controlled_plants.csv",
    "data/controlled_pollinators.csv",
    partition_b_c,
    num_communities_b_c,
    communities_grouped_b_c,
    network_type="controlled",
    min_spacing=0.02,
    min_size=100,
    scale_factor=100)

f.draw_network_communities(
    "data/grestored.csv",
    "data/restored_plants.csv",
    "data/restored_pollinators.csv",
    partition_l_r,
    num_communities_l_r,
    communities_grouped_l_r,
    network_type="restored",
    min_spacing=0.02,
    min_size=100,
    scale_factor=100)

f.draw_network_communities(
    "data/grestored.csv",
    "data/restored_plants.csv",
    "data/restored_pollinators.csv",
    partition_b_r,
    num_communities_b_r,
    communities_grouped_b_r,
    network_type="restored",
    min_spacing=0.02,
    min_size=100,
    scale_factor=100)

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

plant_degrees_c, pollinator_degrees_c, all_degrees_c = degree("data/gcontrolled.csv")
plant_degrees_r, pollinator_degrees_r, all_degrees_r = degree("data/grestored.csv")

print(modularity_b_c, modularity_b_r)
