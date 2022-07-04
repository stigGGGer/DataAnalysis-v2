from enum import Enum
import pandas as pd

from sklearn.cluster import AffinityPropagation

class Affinity(Enum):
    euclidean = "euclidean"
    precomputed = "precomputed"

def Affinity_Propagation(table, parametrs):
    clustering = AffinityPropagation(affinity = parametrs[0], preference = parametrs[1], damping = parametrs[2], max_iter = parametrs[3])
    Y_preds = pd.DataFrame(data = clustering.fit_predict(table), columns = ["Y_Pred"])
    return Y_preds
