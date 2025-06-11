import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import community as co
from sklearn.metrics import adjusted_rand_score, mutual_info_score
import matplotlib.cm as cm
import graph_tool.all as gt
import functions as fu

# --- Main execution part (outside the function) ---
if __name__ == "__main__":
    # Ensure your data files exist for this to run
    # gcontrolled.csv, controlled_sorted.csv, controlled_animal_sorted.csv, grestored.csv
    # and the specific controlled_families CSVs (e.g., gcontrolled_coleoptera.csv etc.)

    pollinators_c, plants_c, adj_matrix_c = fu.load_adjacency_matrix("data/gcontrolled.csv")
    pollinators_r, plants_r, adj_matrix_r = fu.load_adjacency_matrix("data/grestored.csv")

    common_pollinators = set(pollinators_c) & set(pollinators_r)
    common_plants = set(plants_c) & set(plants_r)

    controlled_families = ["data/gcontrolled_coleoptera.csv", "data/gcontrolled_diptera.csv", "data/gcontrolled_hymenoptera.csv", "data/gcontrolled_lepidoptera.csv", "data/gcontrolled_squamata.csv"]
    evenness_c_list = []
    for f in controlled_families:
        # Direct call, will error if file is not found or malformed
        evenness_c_list.append(fu.evenness(f))

    # Condition simplified as there's no "Exception" handling for append
    if evenness_c_list and not all(np.isnan(evenness_c_list)): # This specific check uses np.isnan, which is not an "explicit data existence or validity check" in the same vein as checking for missing files or keys, so it remains.
        print(evenness_c_list)
        print(f"Average evenness for controlled families: {np.nanmean(evenness_c_list):.4f}")
    else:
        print("No valid evenness values calculated for controlled families.")

    # Direct call, will error if file is not found or malformed
    evenness_r = fu.evenness("data/grestored.csv")
    print("Restored evenness:", evenness_r)

    # Direct call, will error if file is not found or malformed
    evenness_c = fu.evenness("data/gcontrolled.csv")
    print("Controlled evenness:", evenness_c)

    # --- Call the modified function and get the graph-tool graph ---
    G_nx_controlled, g_gt_controlled, louvain_partition_controlled = fu.draw_network_communities("data/gcontrolled.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", common_plants, network_type = "controlled", output_txt_file_suffix="_community_details.txt")

    plt.title("Bipartite Pollination Graph - Controlled Network with Communities", fontsize=16)
    plt.savefig("controlled_graph_with_communities.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Test controlled graph-tool graph against BiSBM ---
    print("\n--- Testing Controlled Network against Bipartite Stochastic Block Model ---")
    # CORRECTED LINE: bipartite_nodes passed via state_args
    bisbm_state_controlled = gt.minimize_blockmodel_dl(g_gt_controlled, state_args={'bipartite_nodes': g_gt_controlled.vp.bipartite})
    optimal_bisbm_dl_controlled = bisbm_state_controlled.entropy()
    print(f"Optimal BiSBM Description Length for Controlled Network: {optimal_bisbm_dl_controlled:.4f}")

    # Prepare Louvain partition for comparison with BiSBM
    # Create a graph-tool PropertyMap for Louvain communities from your networkx output
    louvain_blocks_gt_controlled = g_gt_controlled.new_vertex_property("int")
    for node_name, comm_id in louvain_partition_controlled.items():
        v = None
        for temp_v in g_gt_controlled.vertices():
            if g_gt_controlled.vp.name[temp_v] == node_name:
                v = temp_v
                break
        # This line will error if 'v' is None (i.e., node_name not found in graph-tool graph)
        louvain_blocks_gt_controlled[v] = comm_id 

    # Direct calls, will error if there's an issue with the state or entropy calculation
    # CORRECTED LINE: bipartite_nodes passed via state_args
    louvain_state_controlled = gt.BlockState(g_gt_controlled, b=louvain_blocks_gt_controlled, state_args={'bipartite_nodes': g_gt_controlled.vp.bipartite})
    louvain_dl_controlled = louvain_state_controlled.entropy()
    print(f"Description Length of Louvain partition for Controlled Network under BiSBM: {louvain_dl_controlled:.4f}")

    if louvain_dl_controlled < optimal_bisbm_dl_controlled:
        print("Louvain partition is more parsimonious or as parsimonious as BiSBM's optimal fit.")
    elif louvain_dl_controlled == optimal_bisbm_dl_controlled:
        print("Louvain partition matches BiSBM's optimal fit in description length.")
    else:
        print("BiSBM's optimal fit has a lower description length than the Louvain partition, suggesting a more statistically parsimonious structure.")

    # Comparison metrics
    # from sklearn.metrics import adjusted_rand_score, mutual_info_score # Already imported

    # Convert community assignments to arrays for comparison
    # Using a stable order (e.g., iterating through vertices)
    bisbm_labels_controlled = np.array([bisbm_state_controlled.get_blocks()[v] for v in g_gt_controlled.vertices()])
    # For the Controlled Network
    # Retrieve the block assignments as a vertex property map
    bisbm_blocks_vprop = bisbm_state_controlled.get_blocks()

    # Create a dictionary mapping node names to their BiSBM community IDs
    bisbm_partition_controlled = {}
    for v in g_gt_controlled.vertices():
        node_name = g_gt_controlled.vp.name[v] # Get the original node name
        community_id = bisbm_blocks_vprop[v]    # Get its BiSBM community ID
        bisbm_partition_controlled[node_name] = community_id

    print("\n--- BiSBM Communities for Controlled Network ---")
    # You can now print them out or process them
    communities_grouped_bisbm_controlled = {}
    for node, comm_id in bisbm_partition_controlled.items():
        if comm_id not in communities_grouped_bisbm_controlled:
            communities_grouped_bisbm_controlled[comm_id] = []
        communities_grouped_bisbm_controlled[comm_id].append(node)

    for comm_id, nodes_in_comm in communities_grouped_bisbm_controlled.items():
        # You'll need to know which are plants/pollinators from G_nx_controlled.nodes[node]['bipartite']
        plants_in_comm = [n for n in nodes_in_comm if G_nx_controlled.nodes[n]['bipartite'] == 0]
        pollinators_in_comm = [n for n in nodes_in_comm if G_nx_controlled.nodes[n]['bipartite'] == 1]
        print(f"BiSBM Community {comm_id}:")
        print(f"  Plants ({len(plants_in_comm)}): {', '.join(plants_in_comm) if plants_in_comm else 'None'}")
        print(f"  Pollinators ({len(pollinators_in_comm)}): {', '.join(pollinators_in_comm) if pollinators_in_comm else 'None'}")
        print("-" * 30)
        
    louvain_labels_controlled = np.array([louvain_blocks_gt_controlled[v] for v in g_gt_controlled.vertices()])

    ari_controlled = adjusted_rand_score(louvain_labels_controlled, bisbm_labels_controlled)
    # This will error if the denominator is zero (e.g., if one partition has only a single community).
    nmi_denominator_controlled = max(mutual_info_score(louvain_labels_controlled, louvain_labels_controlled), mutual_info_score(bisbm_labels_controlled, bisbm_labels_controlled))
    nmi_controlled = mutual_info_score(louvain_labels_controlled, bisbm_labels_controlled) / nmi_denominator_controlled

    print(f"Adjusted Rand Index (ARI) for Controlled Network: {ari_controlled:.4f}")
    print(f"Normalized Mutual Information (NMI) for Controlled Network: {nmi_controlled:.4f}")

    # --- Process Restored Network ---
    G_nx_restored, g_gt_restored, louvain_partition_restored = \
        fu.draw_network_communities("data/grestored.csv", "data/restored_plants.csv", "data/restored_pollinators.csv", common_plants, network_type = "restored", output_txt_file_suffix="_community_details.txt")

    plt.title("Bipartite Pollination Graph - Restored Network with Communities", fontsize=16)
    plt.savefig("restored_graph_with_communities.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Test restored graph-tool graph against BiSBM ---
    print("\n--- Testing Restored Network against Bipartite Stochastic Block Model ---")
    # CORRECTED LINE: bipartite_nodes passed via state_args
    bisbm_state_restored = gt.minimize_blockmodel_dl(g_gt_restored, state_args={'bipartite_nodes': g_gt_restored.vp.bipartite})
    optimal_bisbm_dl_restored = bisbm_state_restored.entropy()
    print(f"Optimal BiSBM Description Length for Restored Network: {optimal_bisbm_dl_restored:.4f}")

    # Prepare Louvain partition for comparison with BiSBM
    louvain_blocks_gt_restored = g_gt_restored.new_vertex_property("int")
    for node_name, comm_id in louvain_partition_restored.items():
        v = None
        for temp_v in g_gt_restored.vertices():
            if g_gt_restored.vp.name[temp_v] == node_name:
                v = temp_v
                break
        # This line will error if 'v' is None
        louvain_blocks_gt_restored[v] = comm_id

    # Direct calls, will error if there's an issue with the state or entropy calculation
    # CORRECTED LINE: bipartite_nodes passed via state_args
    louvain_state_restored = gt.BlockState(g_gt_restored, b=louvain_blocks_gt_restored, state_args={'bipartite_nodes': g_gt_restored.vp.bipartite})
    louvain_dl_restored = louvain_state_restored.entropy()
    print(f"Description Length of Louvain partition for Restored Network under BiSBM: {louvain_dl_restored:.4f}")

    if louvain_dl_restored < optimal_bisbm_dl_restored:
        print("Louvain partition is more parsimonious or as parsimonious as BiSBM's optimal fit.")
    elif louvain_dl_restored == optimal_bisbm_dl_restored:
        print("Louvain partition matches BiSBM's optimal fit in description length.")
    else:
        print("BiSBM's optimal fit has a lower description length than the Louvain partition, suggesting a more statistically parsimonious structure.")

    # Comparison metrics
    # from sklearn.metrics import adjusted_rand_score, mutual_info_score # Already imported

    bisbm_labels_restored = np.array([bisbm_state_restored.get_blocks()[v] for v in g_gt_restored.vertices()])
    # For the Restored Network
    # Retrieve the block assignments as a vertex property map
    bisbm_blocks_vprop = bisbm_state_restored.get_blocks()

    # Create a dictionary mapping node names to their BiSBM community IDs
    bisbm_partition_restored = {}
    for v in g_gt_restored.vertices():
        node_name = g_gt_restored.vp.name[v] # Get the original node name
        community_id = bisbm_blocks_vprop[v]    # Get its BiSBM community ID
        bisbm_partition_restored[node_name] = community_id

    print("\n--- BiSBM Communities for Restored Network ---")
    # You can now print them out or process them
    communities_grouped_bisbm_restored = {}
    for node, comm_id in bisbm_partition_restored.items():
        if comm_id not in communities_grouped_bisbm_restored:
            communities_grouped_bisbm_restored[comm_id] = []
        communities_grouped_bisbm_restored[comm_id].append(node)

    for comm_id, nodes_in_comm in communities_grouped_bisbm_restored.items():
        # You'll need to know which are plants/pollinators from G_nx_controlled.nodes[node]['bipartite']
        plants_in_comm = [n for n in nodes_in_comm if G_nx_restored.nodes[n]['bipartite'] == 0]
        pollinators_in_comm = [n for n in nodes_in_comm if G_nx_restored.nodes[n]['bipartite'] == 1]
        print(f"BiSBM Community {comm_id}:")
        print(f"  Plants ({len(plants_in_comm)}): {', '.join(plants_in_comm) if plants_in_comm else 'None'}")
        print(f"  Pollinators ({len(pollinators_in_comm)}): {', '.join(pollinators_in_comm) if pollinators_in_comm else 'None'}")
        print("-" * 30)
        
    louvain_labels_restored = np.array([louvain_blocks_gt_restored[v] for v in g_gt_restored.vertices()])

    ari_restored = adjusted_rand_score(louvain_labels_restored, bisbm_labels_restored)
    # This will error if the denominator is zero.
    nmi_denominator_restored = max(mutual_info_score(louvain_labels_restored, louvain_labels_restored), mutual_info_score(bisbm_labels_restored, bisbm_labels_restored))
    nmi_restored = mutual_info_score(louvain_labels_restored, bisbm_labels_restored) / nmi_denominator_restored

    print(f"Adjusted Rand Index (ARI) for Restored Network: {ari_restored:.4f}")
    print(f"Normalized Mutual Information (NMI) for Restored Network: {nmi_restored:.4f}")
            
    df_bisbm_controlled = pd.DataFrame(bisbm_partition_controlled.items(), columns=['Node', 'BiSBM_Community'])
    df_bisbm_controlled.to_csv("controlled_bisbm_communities.csv", index=False)
    print("\nSaved controlled_bisbm_communities.csv")

    df_bisbm_restored = pd.DataFrame(bisbm_partition_restored.items(), columns=['Node', 'BiSBM_Community'])
    df_bisbm_restored.to_csv("restored_bisbm_communities.csv", index=False)
    print("\nSaved restored_bisbm_communities.csv")
    
    
    
    plt.title("Plant-Pollinator Bipartite Network with Detected Communities", fontsize=16)
