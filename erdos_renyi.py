import functions as f

N_ER = 100

# find common plants and pollinators
pollinators_c, plants_c, adj_matrix_c = f.load_adjacency_matrix("data/gcontrolled.csv")
pollinators_r, plants_r, adj_matrix_r = f.load_adjacency_matrix("data/grestored.csv")

common_pollinators = set(pollinators_c) & set(pollinators_r)
common_plants = set(plants_c) & set(plants_r)

f.centrality_measures(N_ER, "data/grestored.csv", common_plants, common_pollinators, graph_type="Controlled")
f.centrality_measures(N_ER, "data/gcontrolled.csv", common_plants, common_pollinators, graph_type="Restored")