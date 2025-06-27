import matplotlib.pyplot as plt
import functions as f

pollinators_c, plants_c, adj_matrix_c = f.load_adjacency_matrix("data/gcontrolled.csv")
pollinators_r, plants_r, adj_matrix_r = f.load_adjacency_matrix("data/grestored.csv")

common_pollinators = set(pollinators_c) & set(pollinators_r)
common_plants = set(plants_c) & set(plants_r)

f.draw_network_order("data/gcontrolled.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", common_plants, min_spacing=0.02, min_size=50, scale_factor=150)
plt.savefig("controlled_graph.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_order("data/grestored.csv", "data/restored_plants.csv", "data/restored_pollinators.csv" ,common_plants, min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("restored_graph.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/gcontrolled.csv", "data/controlled_plants.csv", "data/controlled_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("controlled_graph_origin.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

f.draw_network_origin("data/grestored.csv", "data/restored_plants.csv", "data/restored_pollinators.csv", min_spacing=0.02, min_size=70, scale_factor=150)
plt.savefig("restored_graph_origin.png", format='png', dpi=300, bbox_inches='tight')
plt.show()