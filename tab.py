import pandas as pd
import numpy as np
import functions as f

def print_graph_info(file):
    data = pd.read_csv(file, index_col=0, header=0, encoding='ISO-8859-1')

    #data = data_d.to_numpy()

    num_animals, num_plants = data.shape

    interactions = np.count_nonzero(data)
    
    return num_plants, num_animals, interactions

def number_of_visits(file_path):
    df = pd.read_csv(file_path)

    number_of_visits = df.iloc[:, 6].sum()

    return number_of_visits

# -- Number of plants, animals and interactions --
# control site
plants_controlled, animals_controlled, interactions_controlled = print_graph_info("data/gcontrolled.csv")
print("Number of plant species in control site:", plants_controlled)
print("Number of animal species in control site:", animals_controlled)
print("Number of interactions in control site:", interactions_controlled)
# restored site
plants_restored, animals_restored, interactions_restored = print_graph_info("data/grestored.csv")
print("Number of plant species in restored site:", plants_restored)
print("Number of animal species in restored site:", animals_restored)
print("Number of interactions in restored site:", interactions_restored)

# -- Evenness --
# control site
evenness_c = f.evenness("data/gcontrolled.csv")
print("controlled evenness:", evenness_c)
#restored site
evenness_r = f.evenness("data/grestored.csv")
print("restored evenness:", evenness_r)

# -- Number of visits --
# control site
num_visits_c = number_of_visits("data/controlled_pollinators.csv")
print("Number of visits in controlled site:", num_visits_c)
# restored site
num_visits_r = number_of_visits("data/restored_pollinators.csv")
print("Number of visits in restored site:", num_visits_r)

# -- Maximal and Mean linkage(degree) --
# control site
G_c = f.create_network("data/gcontrolled.csv")
plants_degrees_c, pollinators_degrees_c = f.degree(G_c)
max_plants_degree_c = max(plants_degrees_c)
max_pollinators_degree_c = max(pollinators_degrees_c)
mean_plants_degree_c = np.mean(np.array(plants_degrees_c))
std_dev_plants_degree_c = np.std(np.array(plants_degrees_c))/np.sqrt(len(plants_degrees_c))
mean_pollinators_degree_c = np.mean(np.array(pollinators_degrees_c))
std_dev_pollinators_degree_c = np.std(np.array(pollinators_degrees_c))/np.sqrt(len(pollinators_degrees_c))
print("Maximal plant linkage in controlled site:", max_plants_degree_c)
print("Maximal pollinator linkage in controlled site:", max_pollinators_degree_c)
print(f'Mean plant linkage in controlled site: {mean_plants_degree_c} +/- {std_dev_plants_degree_c}')
print(f'Mean pollinator linkage in controlled site: {mean_pollinators_degree_c} +/- {std_dev_pollinators_degree_c}')
# restored site
G_r = f.create_network("data/grestored.csv")
plants_degrees_r, pollinators_degrees_r = f.degree(G_r)
max_plants_degree_r = max(plants_degrees_r)
max_pollinators_degree_r = max(pollinators_degrees_r)
mean_plants_degree_r = np.mean(np.array(plants_degrees_r))
std_dev_plants_degree_r = np.std(np.array(plants_degrees_r))/np.sqrt(len(plants_degrees_r))
mean_pollinators_degree_r = np.mean(np.array(pollinators_degrees_c))
std_dev_pollinators_degree_r = np.std(np.array(pollinators_degrees_c))/np.sqrt(len(pollinators_degrees_r))
print("Maximal plant linkage in restored site:", max_plants_degree_r)
print("Maximal pollinator linkage in restored site:", max_pollinators_degree_r)
print(f'Mean plant linkage in restored site: {mean_plants_degree_r} +/- {std_dev_plants_degree_r}')
print(f'Mean pollinator linkage in restored site: {mean_pollinators_degree_r} +/- {std_dev_pollinators_degree_r}')
