from enum import Enum

from sklearn.cluster import AffinityPropagation

class Affinity(Enum):
    euclidean = "euclidean"
    precomputed = "precomputed"

def Affinity_Propagation(table, parametrs):
    clustering = AffinityPropagation(random_state = parametrs[1], affinity = parametrs[0], preference = parametrs[2], damping = parametrs[3])
    #clustering = AffinityPropagation(max_iter=500, random_state = parametrs[1], affinity = parametrs[0], damping = parametrs[2])
    Y_preds = clustering.fit_predict(table)
    #return [table, Y_preds]
    return table.assign(Y_Pred=Y_preds)
