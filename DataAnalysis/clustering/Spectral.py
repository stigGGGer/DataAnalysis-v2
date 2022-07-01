import pandas as pd
from sklearn.cluster import SpectralClustering
import numpy as np


def Spectral(table, parametrs):
    clustering = SpectralClustering(n_clusters=3, assign_labels='discretize', random_state=0).fit(table)
    Y_preds = pd.DataFrame(data = clustering.labels_, columns = ["Y_Pred"])
    return Y_preds
