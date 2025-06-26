import pandas as pd
#import igraph as ig
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

    seventh_column_sum = df[6].sum()

    return float(seventh_column_sum) 

# number of plants, animals and interactions
plants_controlled, animals_controlled, interactions_controlled = print_graph_info("data/gcontrolled.csv")
plants_restored, animals_restored, interactions_restored = print_graph_info("data/grestored.csv")
print("Number of plant species in control site:", plants_controlled)
print("Number of animal species in control site:", animals_controlled)
print("Number of interactions in control site:", interactions_controlled)
print("Number of plant species in restored site:", plants_restored)
print("Number of animal species in restored site:", animals_restored)
print("Number of interactions in restored site:", interactions_restored)

# evenness
evenness_r = f.evenness("data/grestored.csv")
evenness_c = f.evenness("data/gcontrolled.csv")
print("controlled evenness:", evenness_c)
print("restored evenness:", evenness_r)

# number of visits
num_visits_c = number_of_visits("data/controlled_pollinators.csv")
num_visits_r = number_of_visits("data/restored_pollinators.csv")
print("Number of visits in controlled site:", num_visits_c)
print("Number of visits in restored site:", num_visits_r)


"""
bisogna calcolare: 
    1. number of visits (a partire dai file delle piante sommando una colonna)
    2. maximal plant linkage (prendere il massimo del degree)
    3. maximal amimal linkage (same)
    4. mean +\- SE plant and animal linkage (fai dai degree)
"""