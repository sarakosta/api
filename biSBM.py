import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import bipartiteSBM
import seaborn as sb

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')
    return df.index.tolist(), df.columns.tolist(), df.values

row_labels, col_labels, adj_matrix = load_adjacency_matrix("gcontrolled.csv")

# model = bipartiteSBM.biSBM(data = adj_matrix, fixed_params = None, n_iters = 1000, random_seed = 42)
# model.run()

print(bipartiteSBM.__file__)