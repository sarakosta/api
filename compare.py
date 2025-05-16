import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')  # First column is row labels
    return df.index.tolist(), df.columns.tolist(), df.values  # Extract row labels, column labels, and matrix

pollinators_c, plants_c, adj_matrix_c = load_adjacency_matrix("gcontrolled.csv")
pollinators_r, plants_r, adj_matrix_r = load_adjacency_matrix("grestored.csv")

common_pollinators = set(pollinators_c) & set(pollinators_r)
common_plants = set(plants_c) &  set(plants_r)
print("Common plants:")
for plant in common_plants:
    print(plant)
print("Number of common plants:", len(common_plants))
    
print("Common pollinators:")
for pollinator in common_pollinators:
    print(pollinator)
print("Number of common pollinators:" ,len(common_pollinators))