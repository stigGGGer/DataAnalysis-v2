from enum import Enum
import pandas as pd

from sklearn.cluster import AffinityPropagation

class Affinity(Enum):
    euclidean = "euclidean"
    precomputed = "precomputed"

def Affinity_Propagation(table, parametrs):
    clustering = AffinityPropagation(random_state = parametrs[1], affinity = parametrs[0], preference = parametrs[2], damping = parametrs[3], max_iter = parametrs[4])
    Y_preds = pd.DataFrame(data = clustering.fit_predict(table), columns = ["Y_Pred"])
    return Y_preds
