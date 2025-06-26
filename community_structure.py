import matplotlib.pyplot as plt
import functions as f

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

print("Modularities:", modularity_b_c, modularity_b_r, modularity_plants_c, modularity_pollinators_c, modularity_plants_r, modularity_pollinators_r)
