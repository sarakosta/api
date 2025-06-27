import matplotlib.pyplot as plt
import functions as f

# define controlled graph
G_c = f.create_network("data/gcontrolled.csv")
# community detection
# partition_l_c, num_communities_l_c, modularity_l_c, communities_grouped_l_c = f.print_Louvain_communities(G_c, N_louvain=1000, network_type="controlled")
# partition_b_c, num_communities_b_c, optimal_bisbm_dl_c, communities_grouped_b_c, modularity_b_c, _ = f.print_biSBM_communities(G_c, N_biSBM=1000, network_type="controlled")
# partition_plants_c, num_communities_plants_c, optimal_sbm_dl_plants_c, communities_grouped_plants_c, modularity_plants_c, state_plants_unipartite_c, partition_pollinators_c, num_communities_plants_c, optimal_sbm_dl_pollinators_c, communities_grouped_pollinators_c, modularity_pollinators_c, state_pollinators_unipartite_c = f.print_SBM_communities(G_c, N_SBM=1000, network_type="controlled")

# define restored graph
G_r = f.create_network("data/grestored.csv")
# community detection
# partition_l_r, num_communities_l_r, modularity_l_r, communities_grouped_l_r = f.print_Louvain_communities(G_r, N_louvain=1000, network_type="restored")
# partition_b_r, num_communities_b_r, optimal_bisbm_dl_r, communities_grouped_b_r, modularity_b_r, _ = f.print_biSBM_communities(G_r, N_biSBM=1000, network_type="restored")
# partition_plants_r, num_communities_plants_r, optimal_sbm_dl_plants_r, communities_grouped_plants_r, modularity_plants_r, state_plants_unipartite_r, partition_pollinators_r, num_communities_plants_r, optimal_sbm_dl_pollinators_r, communities_grouped_pollinators_r, modularity_pollinators_r, state_pollinators_unipartite_r = f.print_SBM_communities(G_r, N_SBM=1000, network_type="restored")

"""
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
"""

# Louvain community detection on projected graphs
G_plants_c, G_pollinators_c = f.projections(G_c)
partition_l_plants_c, num_communities_l_plants_c, modularity_l_plants_c, communities_grouped_l_plants_c = f.print_projected_Louvain_communities(G_plants_c, N_louvain = 1000, network_type = "controlled", projection_type = "plants")
partition_l_pollinators_c, num_communities_l_pollinators_c, modularity_l_pollinators_c, communities_grouped_l_pollinators_c = f.print_projected_Louvain_communities(G_pollinators_c, N_louvain = 1000, network_type = "controlled", projection_type = "pollinators")

G_plants_r, G_pollinators_r = f.projections(G_r)
partition_l_plants_r, num_communities_l_plants_r, modularity_l_plants_r, communities_grouped_l_plants_r = f.print_projected_Louvain_communities(G_plants_r, N_louvain = 1000, network_type = "restored", projection_type = "plants")
partition_l_pollinators_r, num_communities_l_pollinators_r, modularity_l_pollinators_r, communities_grouped_l_pollinators_r = f.print_projected_Louvain_communities(G_pollinators_r, N_louvain = 1000, network_type = "restored", projection_type = "pollinators")

f.draw_network_communities_projected(
    G_plants_c,
    partition_l_plants_c,
    num_communities_l_plants_c,
    communities_grouped_l_plants_c,
    network_type = "controlled",
    projection_type="plants",
    comm_detection_method = "louvain"
)
plt.savefig("controlled_graph_louvain_communities_plantproj.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_communities_projected(
    G_pollinators_c,
    partition_l_pollinators_c,
    num_communities_l_pollinators_c,
    communities_grouped_l_pollinators_c,
    network_type = "controlled",
    projection_type="pollinators",
    comm_detection_method = "louvain"
)
plt.savefig("controlled_graph_louvain_communities_pollproj.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_communities_projected(
    G_plants_r,
    partition_l_plants_r,
    num_communities_l_plants_r,
    communities_grouped_l_plants_r,
    network_type = "restored",
    projection_type="plants",
    comm_detection_method = "louvain"
)
plt.savefig("restored_graph_louvain_communities_plantproj.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_communities_projected(
    G_pollinators_r,
    partition_l_pollinators_r,
    num_communities_l_pollinators_r,
    communities_grouped_l_pollinators_r,
    network_type = "restored",
    projection_type="pollinators",
    comm_detection_method = "louvain"
)
plt.savefig("restored_graph_louvain_communities_pollproj.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

"""
print("Modularity Louvain Control:", modularity_l_c)
print("Modularity Louvain Restored:", modularity_l_r)
print("Modularity biSBM Control:", modularity_b_c)
print("Modularity biSBM Restored:", modularity_b_r)
print("Modularity Louvain Control:", modularity_l_c)
print("Modularity Plant Projection SBM Control:", modularity_plants_c)
print("Modularity Plant Projection SBM Restored:", modularity_plants_r)
print("Modularity Pollinator Projection SBM Control:", modularity_pollinators_c)
print("Modularity Pollinator Projection SBM Restored:", modularity_pollinators_r)
"""