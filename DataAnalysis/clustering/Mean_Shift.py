from sklearn.cluster import MeanShift
import numpy as np
import pandas as pd


def Mean_Shift(table, parametrs):
    clustering = MeanShift(bandwidth=2).fit(table)
    Y_preds = pd.DataFrame(data = clustering.labels_, columns = ["Y_Pred"])
    return Y_preds