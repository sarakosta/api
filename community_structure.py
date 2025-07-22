import matplotlib.pyplot as plt
import functions as f

# define controlled graph
G_c = f.create_network("data/gcontrolled.csv")
# community detection
partition_l_c, num_communities_l_c, modularity_l_c, communities_grouped_l_c = f.print_Louvain_communities(G_c, N_louvain=1000, network_type="controlled")

# define restored graph
G_r = f.create_network("data/grestored.csv")
# community detection
partition_l_r, num_communities_l_r, modularity_l_r, communities_grouped_l_r = f.print_Louvain_communities(G_r, N_louvain=1000, network_type="restored")

f.draw_network_communities(
    "data/gcontrolled.csv",
    "data/controlled_plants.csv",
    "data/controlled_pollinators.csv",
    partition_l_c,
    num_communities_l_c,
    communities_grouped_l_c,
    network_type="controlled",
    comm_detection_method="louvain",
    min_spacing=0.02,
    min_size=100,
    scale_factor=100)
plt.savefig("controlled_graph_louvain_communities.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_communities(
    "data/grestored.csv",
    "data/restored_plants.csv",
    "data/restored_pollinators.csv",
    partition_l_r,
    num_communities_l_r,
    communities_grouped_l_r,
    network_type="restored",
    comm_detection_method="louvain",
    min_spacing=0.02,
    min_size=100,
    scale_factor=100)
plt.savefig("restored_graph_louvain_communities.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()


# Louvain community detection on projected graphs
G_plants_c, G_pollinators_c = f.projections(G_c)
partition_l_plants_c, num_communities_l_plants_c, modularity_l_plants_c, communities_grouped_l_plants_c = f.print_projected_Louvain_communities(G_plants_c, N_louvain = 1000, network_type = "controlled", projection_type = "plants")
partition_l_pollinators_c, num_communities_l_pollinators_c, modularity_l_pollinators_c, communities_grouped_l_pollinators_c = f.print_projected_Louvain_communities(G_pollinators_c, N_louvain = 1000, network_type = "controlled", projection_type = "pollinators")

G_plants_r, G_pollinators_r = f.projections(G_r)
partition_l_plants_r, num_communities_l_plants_r, modularity_l_plants_r, communities_grouped_l_plants_r = f.print_projected_Louvain_communities(G_plants_r, N_louvain = 1000, network_type = "restored", projection_type = "plants")
partition_l_pollinators_r, num_communities_l_pollinators_r, modularity_l_pollinators_r, communities_grouped_l_pollinators_r = f.print_projected_Louvain_communities(G_pollinators_r, N_louvain = 1000, network_type = "restored", projection_type = "pollinators")