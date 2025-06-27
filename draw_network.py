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



"""
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
"""
