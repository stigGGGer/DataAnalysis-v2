import numpy as np
from enum import Enum
import pandas as pd
from sklearn.cluster import Birch


def myBirch(table, parametrs):
    brc = Birch(n_clusters=None).fit(table)
    Y_preds = pd.DataFrame(data = brc, columns = ["Y_Pred"])
    return Y_preds
