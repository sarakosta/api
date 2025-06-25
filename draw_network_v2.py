#import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import functions as f
import graph_tool.all as gt
import csv

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

# define controlled graph
G_c = f.create_network("data/gcontrolled.csv")
# community detection
partition_l_c, num_communities_l_c, modularity_c, communities_grouped_l_c = f.print_Louvain_communities(G_c, N_louvain=1000, network_type="controlled")
partition_b_c, num_communities_b_c, optimal_bisbm_dl_c, communities_grouped_b_c, modularity_b_c, _ = f.print_biSBM_communities(G_c, N_biSBM=1000, network_type="controlled")
partition_plants_c, num_communities_plants_c, optimal_sbm_dl_plants_c, communities_grouped_plants_c, modularity_plants_c, state_plants_unipartite_c, partition_pollinators_c, num_communities_plants_c, optimal_sbm_dl_pollinators_c, communities_grouped_pollinators_c, modularity_pollinators_c, state_pollinators_unipartite_c = f.print_SBM_communities(G_c, N_SBM=1000, network_type="controlled")


# define restored graph
G_r = f.create_network("data/grestored.csv")
# community detection
partition_l_r, num_communities_l_r, modularity_r, communities_grouped_l_r = f.print_Louvain_communities(G_r, N_louvain=1000, network_type="restored")
partition_b_r, num_communities_b_r, optimal_bisbm_dl_r, communities_grouped_b_r, modularity_b_r, _ = f.print_biSBM_communities(G_r, N_biSBM=1000, network_type="restored")
partition_plants_r, num_communities_plants_r, optimal_sbm_dl_plants_r, communities_grouped_plants_r, modularity_plants_r, state_plants_unipartite_r, partition_pollinators_r, num_communities_plants_r, optimal_sbm_dl_pollinators_r, communities_grouped_pollinators_r, modularity_pollinators_r, state_pollinators_unipartite_r = f.print_SBM_communities(G_r, N_SBM=1000, network_type="restored")


"""
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
"""

"""
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
"""

print("Modularities:", modularity_b_c, modularity_b_r, modularity_plants_c, modularity_pollinators_c, modularity_plants_r, modularity_pollinators_r)

G_plants_c, G_pollinators_c = f.projections(G_c)
adj_matrix_plants_c = nx.adjacency_matrix(G_plants_c)
adj_matrix_plants_c = adj_matrix_plants_c.toarray()

# Define the filename
filename = 'adjacency_matrix_csv_module.csv'

# Open the file in write mode
with open(filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Optional: Write a header row if you have vertex labels
    # For example, if your vertices are 'V0', 'V1', 'V2', 'V3'
    # vertices = ['V0', 'V1', 'V2', 'V3']
    # csv_writer.writerow([''] + vertices) # Empty string for the top-left cell

    # Write each row of the adjacency matrix
    for row in adj_matrix_plants_c:
        csv_writer.writerow(row)
        # If you wanted to include vertex labels as the first column for each row:
        # vertex_label_for_this_row = 'V' + str(adj_matrix.index(row)) # Example
        # csv_writer.writerow([vertex_label_for_this_row] + row)


print(f"Adjacency matrix saved to {filename}")
