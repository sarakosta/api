import matplotlib.pyplot as plt
import functions as f

"""
f.draw_network_origin("data/controlled_louvain_community_0_adjacency_matrix.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("controlled_comm0_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/controlled_louvain_community_1_adjacency_matrix.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("controlled_comm1_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/controlled_louvain_community_5_adjacency_matrix.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("controlled_comm5_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/controlled_louvain_community_7_adjacency_matrix.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("controlled_comm7_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/controlled_louvain_community_10_adjacency_matrix.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("controlled_comm10_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/restored_louvain_community_0_adjacency_matrix.csv", "data/restored_plants.csv", "data/restored_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("restored_comm0_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/restored_louvain_community_1_adjacency_matrix.csv", "data/restored_plants.csv", "data/restored_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("restored_comm1_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/restored_louvain_community_2_adjacency_matrix.csv", "data/restored_plants.csv", "data/restored_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("restored_comm2_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/restored_louvain_community_3_adjacency_matrix.csv", "data/restored_plants.csv", "data/restored_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("restored_comm3_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/restored_louvain_community_4_adjacency_matrix.csv", "data/restored_plants.csv", "data/restored_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("restored_comm4_louvain_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()
"""

# controlled plants
f.draw_plants_origin("data/controlled_louvain_plants_community_0_adjacency_matrix.csv", "data/controlled_plants.csv")
plt.savefig("controlled_comm0_louvain_plantproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_plants_origin("data/controlled_louvain_plants_community_1_adjacency_matrix.csv", "data/controlled_plants.csv")
plt.savefig("controlled_comm1_louvain_plantproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_plants_origin("data/controlled_louvain_plants_community_2_adjacency_matrix.csv", "data/controlled_plants.csv")
plt.savefig("controlled_comm2_louvain_plantproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_plants_origin("data/controlled_louvain_plants_community_4_adjacency_matrix.csv", "data/controlled_plants.csv")
plt.savefig("controlled_comm4_louvain_plantproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_plants_origin("data/controlled_louvain_plants_community_5_adjacency_matrix.csv", "data/controlled_plants.csv")
plt.savefig("controlled_comm5_louvain_plantproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

# controlled pollinators
f.draw_pollinators_order("data/controlled_louvain_pollinators_community_0_adjacency_matrix.csv", "data/controlled_pollinators.csv")
plt.savefig("controlled_comm0_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_pollinators_order("data/controlled_louvain_pollinators_community_1_adjacency_matrix.csv", "data/controlled_pollinators.csv")
plt.savefig("controlled_comm1_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_pollinators_order("data/controlled_louvain_pollinators_community_2_adjacency_matrix.csv", "data/controlled_pollinators.csv")
plt.savefig("controlled_comm2_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_pollinators_order("data/controlled_louvain_pollinators_community_3_adjacency_matrix.csv", "data/controlled_pollinators.csv")
plt.savefig("controlled_comm3_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_pollinators_order("data/controlled_louvain_pollinators_community_4_adjacency_matrix.csv", "data/controlled_pollinators.csv")
plt.savefig("controlled_comm4_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

# restored plants
f.draw_plants_origin("data/restored_louvain_plants_community_0_adjacency_matrix.csv", "data/restored_plants.csv")
plt.savefig("restored_comm0_louvain_plantproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_plants_origin("data/restored_louvain_plants_community_2_adjacency_matrix.csv", "data/restored_plants.csv")
plt.savefig("restored_comm2_louvain_plantproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

# restored pollinators
f.draw_pollinators_order("data/restored_louvain_pollinators_community_0_adjacency_matrix.csv", "data/restored_pollinators.csv")
plt.savefig("restored_comm0_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_pollinators_order("data/restored_louvain_pollinators_community_1_adjacency_matrix.csv", "data/restored_pollinators.csv")
plt.savefig("restored_comm1_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_pollinators_order("data/restored_louvain_pollinators_community_2_adjacency_matrix.csv", "data/restored_pollinators.csv")
plt.savefig("restored_comm2_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_pollinators_order("data/restored_louvain_pollinators_community_3_adjacency_matrix.csv", "data/restored_pollinators.csv")
plt.savefig("restored_comm3_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()

f.draw_pollinators_order("data/restored_louvain_pollinators_community_4_adjacency_matrix.csv", "data/restored_pollinators.csv")
plt.savefig("restored_comm4_louvain_pollproj_graph.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
plt.show()


