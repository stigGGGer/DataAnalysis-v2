import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np


def DBScan(table, parametrs):
    clustering = DBSCAN(eps=parametrs[0], min_samples=parametrs[1]).fit(table)
    Y_preds = pd.DataFrame(data = clustering.labels_, columns = ["Y_Pred"])
    return Y_preds
