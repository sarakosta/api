import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import community as co
from sklearn.metrics import adjusted_rand_score, mutual_info_score
import matplotlib.cm as cm
import graph_tool.all as gt
import functions as f

# --- Main execution part (outside the function) ---
if __name__ == "__main__":
    # Ensure your data files exist for this to run
    # gcontrolled.csv, controlled_sorted.csv, controlled_animal_sorted.csv, grestored.csv
    # and the specific controlled_families CSVs (e.g., gcontrolled_coleoptera.csv etc.)

    pollinators_c, plants_c, adj_matrix_c = f.load_adjacency_matrix("data/gcontrolled.csv")
    pollinators_r, plants_r, adj_matrix_r = f.load_adjacency_matrix("data/grestored.csv")

    common_pollinators = set(pollinators_c) & set(pollinators_r)
    common_plants = set(plants_c) & set(plants_r)

    controlled_families = ["data/gcontrolled_coleoptera.csv", "data/gcontrolled_diptera.csv", "data/gcontrolled_hymenoptera.csv", "data/gcontrolled_lepidoptera.csv", "data/gcontrolled_squamata.csv"]
    evenness_c_list = []
    for family in controlled_families:
        # Direct call, will error if file is not found or malformed
        evenness_c_list.append(f.evenness(family))

    # Condition simplified as there's no "Exception" handling for append
    if evenness_c_list and not all(np.isnan(evenness_c_list)): # This specific check uses np.isnan, which is not an "explicit data existence or validity check" in the same vein as checking for missing files or keys, so it remains.
        print(evenness_c_list)
        print(f"Average evenness for controlled families: {np.nanmean(evenness_c_list):.4f}")
    else:
        print("No valid evenness values calculated for controlled families.")

    G_c = f.create_network("data/gcontrolled.csv")
    partition_l_c, num_communities_l_c, modularity_c, communities_grouped_l_c = f.print_Louvain_communities(G_c, N_louvain=1000, network_type="controlled")
    partition_b_c, num_communities_b_c, optimal_bisbm_dl_c, communities_grouped_b_c, modularity_b_c, bisbm_state_c = f.print_biSBM_communities(G_c, N_biSBM=1000, network_type="controlled")

    # --- Call the modified function and get the graph-tool graph ---
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
    plt.title("Bipartite Pollination Graph - Controlled Network with Louvain Communities", fontsize=16)
    plt.savefig("controlled_graph_with_Louvain_communities.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.show()

    g_gt_controlled = f.graph_gt(G_c)
    # Prepare Louvain partition for comparison with BiSBM
    # Create a graph-tool PropertyMap for Louvain communities from your networkx output
    louvain_blocks_gt_controlled = g_gt_controlled.new_vertex_property("int")
    for node_name, comm_id in partition_l_c.items():
        v = None
        for temp_v in g_gt_controlled.vertices():
            if g_gt_controlled.vp.name[temp_v] == node_name:
                v = temp_v
                break
        # This line will error if 'v' is None (i.e., node_name not found in graph-tool graph)
        louvain_blocks_gt_controlled[v] = comm_id 

    # Direct calls, will error if there's an issue with the state or entropy calculation
    louvain_state_controlled = gt.BlockState(g_gt_controlled, b=louvain_blocks_gt_controlled, state_args={'bipartite_nodes': g_gt_controlled.vp.bipartite})
    louvain_dl_controlled = louvain_state_controlled.entropy()
    print(f"Description Length of Louvain partition for Controlled Network under BiSBM: {louvain_dl_controlled:.4f}")

    if louvain_dl_controlled < optimal_bisbm_dl_c:
        print("Louvain partition is more parsimonious or as parsimonious as BiSBM's optimal fit.")
    elif louvain_dl_controlled == optimal_bisbm_dl_c:
        print("Louvain partition matches BiSBM's optimal fit in description length.")
    else:
        print("BiSBM's optimal fit has a lower description length than the Louvain partition, suggesting a more statistically parsimonious structure.")

    # Convert community assignments to arrays for comparison
    bisbm_labels_controlled = np.array([bisbm_state_c.get_blocks()[v] for v in g_gt_controlled.vertices()])
        
    louvain_labels_controlled = np.array([louvain_blocks_gt_controlled[v] for v in g_gt_controlled.vertices()])

    ari_controlled = adjusted_rand_score(louvain_labels_controlled, bisbm_labels_controlled)
    # This will error if the denominator is zero (e.g., if one partition has only a single community).
    nmi_denominator_controlled = max(mutual_info_score(louvain_labels_controlled, louvain_labels_controlled), mutual_info_score(bisbm_labels_controlled, bisbm_labels_controlled))
    nmi_controlled = mutual_info_score(louvain_labels_controlled, bisbm_labels_controlled) / nmi_denominator_controlled

    print(f"Adjusted Rand Index (ARI) for Controlled Network: {ari_controlled:.4f}")
    print(f"Normalized Mutual Information (NMI) for Controlled Network: {nmi_controlled:.4f}")

    # --- RESTORED SITE ---

    G_r = f.create_network("data/grestored.csv")
    partition_l_r, num_communities_l_r, modularity_r, communities_grouped_l_r = f.print_Louvain_communities(G_r, N_louvain=1000, network_type="restored")
    partition_b_r, num_communities_b_r, optimal_bisbm_dl_r, communities_grouped_b_r, modularity_b_r, bisbm_state_c = f.print_biSBM_communities(G_r, N_biSBM=1000, network_type="restored")

    # --- Call the modified function and get the graph-tool graph ---
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
    plt.title("Bipartite Pollination Graph - Restored Network with Louvain Communities", fontsize=16)
    plt.savefig("restored_graph_with_Louvain_communities.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.show()

    g_gt_restored = f.graph_gt(G_r)
    # Prepare Louvain partition for comparison with BiSBM
    # Create a graph-tool PropertyMap for Louvain communities from your networkx output
    louvain_blocks_gt_restored = g_gt_restored.new_vertex_property("int")
    for node_name, comm_id in partition_l_r.items():
        v = None
        for temp_v in g_gt_restored.vertices():
            if g_gt_restored.vp.name[temp_v] == node_name:
                v = temp_v
                break
        # This line will error if 'v' is None (i.e., node_name not found in graph-tool graph)
        louvain_blocks_gt_restored[v] = comm_id 

    # Direct calls, will error if there's an issue with the state or entropy calculation
    louvain_state_restored = gt.BlockState(g_gt_restored, b=louvain_blocks_gt_restored, state_args={'bipartite_nodes': g_gt_restored.vp.bipartite})
    louvain_dl_restored = louvain_state_restored.entropy()
    print(f"Description Length of Louvain partition for Restored Network under BiSBM: {louvain_dl_controlled:.4f}")

    if louvain_dl_restored < optimal_bisbm_dl_r:
        print("Louvain partition is more parsimonious or as parsimonious as BiSBM's optimal fit.")
    elif louvain_dl_controlled == optimal_bisbm_dl_c:
        print("Louvain partition matches BiSBM's optimal fit in description length.")
    else:
        print("BiSBM's optimal fit has a lower description length than the Louvain partition, suggesting a more statistically parsimonious structure.")

    # Convert community assignments to arrays for comparison
    bisbm_labels_restored = np.array([bisbm_state_c.get_blocks()[v] for v in g_gt_restored.vertices()])
        
    louvain_labels_restored = np.array([louvain_blocks_gt_restored[v] for v in g_gt_restored.vertices()])

    ari_restored = adjusted_rand_score(louvain_labels_restored, bisbm_labels_restored)
    # This will error if the denominator is zero (e.g., if one partition has only a single community).
    nmi_denominator_restored = max(mutual_info_score(louvain_labels_restored, louvain_labels_restored), mutual_info_score(bisbm_labels_restored, bisbm_labels_restored))
    nmi_restored = mutual_info_score(louvain_labels_restored, bisbm_labels_restored) / nmi_denominator_restored

    print(f"Adjusted Rand Index (ARI) for Restored Network: {ari_restored:.4f}")
    print(f"Normalized Mutual Information (NMI) for Restored Network: {nmi_restored:.4f}")
            
    #plt.title("Plant-Pollinator Bipartite Network with Detected Communities", fontsize=16)
