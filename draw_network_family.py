#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:19:18 2025

@author: danielecristani
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0, header=0, encoding='ISO-8859-1')  # First column is row labels
    return df.index.tolist(), df.columns.tolist(), df.values

def family_adj_matrix(file_path, file_path2):
    # Load adjacency matrix from CSV with headers
    row_labels, col_labels, adj_matrix = load_adjacency_matrix(file_path)

    # Define node sets
    pollinators = row_labels  # Use actual plant names from CSV headers
    plants = col_labels  # Use actual pollinator names from CSV headers
    
    animal_families = pd.read_csv(file_path2, encoding='ISO-8859-1')
    abundance_value = floral_abundance.loc[floral_abundance.iloc[:, 2] == plant, floral_abundance.columns[7]].values
    
    
    families = ["Coleoptera", "Diptera", "Hemiptera", "Hymenoptera", "Lepidoptera", "Passeriformes", "Squamata"]
    
    adj_matrix_families = [[[]]]
    
    for family in families:
        

    
    