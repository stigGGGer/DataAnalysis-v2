import numpy as np
from enum import Enum
from sklearn.cluster import KMeans

    
def K_Means(table, parametrs):
    kmeans = KMeans(n_clusters=parametrs[0], random_state=parametrs[1]).fit(table)    
    return table.assign(Y_Pred=kmeans.labels_)

