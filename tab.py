import pandas as pd
#import igraph as ig
import numpy as np

def print_graph_info(file):
    data = pd.read_csv(file, index_col=0, header=0, encoding='ISO-8859-1')

    #data = data_d.to_numpy()

    num_animals, num_plants = data.shape

    interactions = np.count_nonzero(data)
    
    return num_plants, num_animals, interactions

plants_controlled, animals_controlled, interactions_controlled = print_graph_info("gcontrolled.csv")
print("Number of plant species in control site:", plants_controlled)
print("Number of animal species in control site:", animals_controlled)
print("Number of interactions in control site:", interactions_controlled)

plants_restored, animals_restored, interactions_restored = print_graph_info("grestored.csv")
print("Number of plant species in restored site:", plants_restored)
print("Number of animal species in restored site:", animals_restored)
print("Number of interactions in restored site:", interactions_restored)