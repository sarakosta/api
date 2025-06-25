import functions as f

# _c = object related to control graph, _r = object related to restored graph

N_ER = 1000

# find common plants and pollinators
pollinators_c, plants_c, adj_matrix_c = f.load_adjacency_matrix("data/gcontrolled.csv")
pollinators_r, plants_r, adj_matrix_r = f.load_adjacency_matrix("data/grestored.csv")

common_pollinators = set(pollinators_c) & set(pollinators_r)
common_plants = set(plants_c) & set(plants_r)

# degree distribution and verification that it is not gaussian
G_c = f.create_network("data/gcontrolled.csv")
plant_degrees_c, pollinator_degrees_c = f.degree(G_c)
G_r = f.create_network("data/grestored.csv")
plant_degrees_r, pollinator_degrees_r = f.degree(G_r)
    
f.gaussian_fit_histo(plant_degrees_c, data_name = 'Controlled', kingdom_name = 'Plants', histo_color = '#1f77b4')
f.gaussian_fit_histo(pollinator_degrees_c, data_name = 'Controlled', kingdom_name = 'Pollinators', histo_color = '#1f77b4')
f.gaussian_fit_histo(plant_degrees_r, data_name = 'Restored', kingdom_name = 'Plants', histo_color = '#ff7f0e')
f.gaussian_fit_histo(pollinator_degrees_r, data_name = 'Restored', kingdom_name = 'Pollinators', histo_color = '#ff7f0e')

# -- Computation of centrality measures for N_ER Erdos-Renyi graphs --
# restored graph as model
mean_bc_plants_c, mean_bc_pollinators_c, mean_cc_plants_c, mean_cc_pollinators_c, mean_wd_plants_c, mean_wd_pollinators_c = f.centrality_measures_ER(N_ER, "data/gcontrolled.csv")
f.bar_chart_ER(plants_c, mean_bc_plants_c, cm_name = "Betweennes Centrality", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_ER(pollinators_c, mean_bc_pollinators_c, cm_name = "Betweennes Centrality", kingdom_type = "Pollinators", graph_type = "Control")
f.bar_chart_ER(plants_c, mean_cc_plants_c, cm_name = "Closeness Centrality", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_ER(pollinators_c, mean_cc_pollinators_c, cm_name = "Closeness Centrality", kingdom_type = "Pollinators", graph_type = "Control")
f.bar_chart_ER(plants_c, mean_wd_plants_c, cm_name = "Weighted Degree", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_ER(pollinators_c, mean_wd_pollinators_c, cm_name = "Weighted Degree", kingdom_type = "Pollinators", graph_type = "Control")

# control graph as model
mean_bc_plants_r, mean_bc_pollinators_r, mean_cc_plants_r, mean_cc_pollinators_r, mean_wd_plants_r, mean_wd_pollinators_r = f.centrality_measures_ER(N_ER, "data/grestored.csv")
f.bar_chart_ER(plants_r, mean_bc_plants_r, cm_name = "Betweennes Centrality", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_ER(pollinators_r, mean_bc_pollinators_r, cm_name = "Betweennes Centrality", kingdom_type = "Pollinators", graph_type = "Restored")
f.bar_chart_ER(plants_r, mean_cc_plants_r, cm_name = "Closeness Centrality", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_ER(pollinators_r, mean_cc_pollinators_r, cm_name = "Closeness Centrality", kingdom_type = "Pollinators", graph_type = "Restored")
f.bar_chart_ER(plants_r, mean_wd_plants_r, cm_name = "Weighted Degree", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_ER(pollinators_r, mean_wd_pollinators_r, cm_name = "Weighted Degree", kingdom_type = "Pollinators", graph_type = "Restored")


# -- Centrality measures for our graphs --
# betweenness centrality
bc_plants_c, bc_pollinators_c = f.betweenness_centrality(G_c)
bc_plants_r, bc_pollinators_r = f.betweenness_centrality(G_r)
f.bar_chart_common_names(bc_plants_c, common_plants, cm_name = "Betweenness Centrality", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_common_names(bc_pollinators_c, common_pollinators, cm_name = "Betweenness Centrality", kingdom_type = "Pollinators", graph_type = "Control")
f.bar_chart_common_names(bc_plants_r, common_plants, cm_name = "Betweenness Centrality", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_common_names(bc_pollinators_r, common_pollinators, cm_name = "Betweenness Centrality", kingdom_type = "Pollinators", graph_type = "Restored")
f.bar_chart_species_origin(bc_plants_c, file_path = "data/controlled_plants.csv", cm_name = "Betweenness Centrality", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_species_origin(bc_pollinators_c, file_path = "data/controlled_pollinators.csv", cm_name = "Betweenness Centrality", kingdom_type = "Pollinators", graph_type = "Control")
f.bar_chart_species_origin(bc_plants_r, file_path = "data/restored_plants.csv", cm_name = "Betweenness Centrality", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_species_origin(bc_pollinators_r, file_path = "data/restored_pollinators.csv", cm_name = "Betweenness Centrality", kingdom_type = "Pollinators", graph_type = "Restored")

# closeness centrality
cc_plants_c, cc_pollinators_c = f.closeness_centrality(G_c)
cc_plants_r, cc_pollinators_r = f.closeness_centrality(G_r)
f.bar_chart_common_names(cc_plants_c, common_plants, cm_name = "Closeness Centrality", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_common_names(cc_pollinators_c, common_pollinators, cm_name = "Closeness Centrality", kingdom_type = "Pollinators", graph_type = "Control")
f.bar_chart_common_names(cc_plants_r, common_plants, cm_name = "Closeness Centrality", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_common_names(cc_pollinators_r, common_pollinators, cm_name = "Closeness Centrality", kingdom_type = "Pollinators", graph_type = "Restored")
f.bar_chart_species_origin(cc_plants_c, file_path = "data/controlled_plants.csv", cm_name = "Closeness Centrality", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_species_origin(cc_pollinators_c, file_path = "data/controlled_pollinators.csv", cm_name = "Closeness Centrality", kingdom_type = "Pollinators", graph_type = "Control")
f.bar_chart_species_origin(cc_plants_r, file_path = "data/restored_plants.csv", cm_name = "Closeness Centrality", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_species_origin(cc_pollinators_r, file_path = "data/restored_pollinators.csv", cm_name = "Closeness Centrality", kingdom_type = "Pollinators", graph_type = "Restored")

# weighted degree 
wd_plants_c, wd_pollinators_c = f.weighted_degree(G_c)
wd_plants_r, wd_pollinators_r = f.weighted_degree(G_r)
f.bar_chart_common_names(wd_plants_c, common_plants, cm_name = "Weighted Degree", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_common_names(wd_pollinators_c, common_pollinators, cm_name = "Weighted Degree", kingdom_type = "Pollinators", graph_type = "Control")
f.bar_chart_common_names(wd_plants_r, common_plants, cm_name = "Weighted Degree", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_common_names(wd_pollinators_r, common_pollinators, cm_name = "Weighted Degree", kingdom_type = "Pollinators", graph_type = "Restored")
f.bar_chart_species_origin(wd_plants_c, file_path = "data/controlled_plants.csv", cm_name = "Weighted Degree", kingdom_type = "Plants", graph_type = "Control")
f.bar_chart_species_origin(wd_pollinators_c, file_path = "data/controlled_pollinators.csv", cm_name = "Weighted Degree", kingdom_type = "Pollinators", graph_type = "Control")
f.bar_chart_species_origin(wd_plants_r, file_path = "data/restored_plants.csv", cm_name = "Weighted Degree", kingdom_type = "Plants", graph_type = "Restored")
f.bar_chart_species_origin(wd_pollinators_r, file_path = "data/restored_pollinators.csv", cm_name = "Weighted Degree", kingdom_type = "Pollinators", graph_type = "Restored")

# -- Mann-Withney and Kolmogorov-Smirnov tests
f.test_ks_mw(list(wd_plants_c.values()), mean_wd_plants_c, list(bc_plants_c.values()), mean_bc_plants_c, list(cc_plants_c.values()), mean_cc_plants_c, graph_type = "Control")
f.test_ks_mw(list(wd_plants_r.values()), mean_wd_plants_r, list(bc_plants_r.values()), mean_bc_plants_r, list(cc_plants_r.values()), mean_cc_plants_r, graph_type = "Restored")

# -- Degree side by side --
plant_degrees_c, pollinator_degrees_c = f.degree(G_c)
plant_degrees_r, pollinator_degrees_r = f.degree(G_r)
# draw the histogram for the degree side by side
f.histo_side_by_side(plant_degrees_c, plant_degrees_r, data_name="Linkage", kingdom_name="Plants")
f.histo_side_by_side(pollinator_degrees_c, pollinator_degrees_r, data_name="Linkage", kingdom_name="Pollinators")
