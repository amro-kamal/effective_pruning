import sys
path_to_add = "../"
sys.path.append(path_to_add)

import numpy as np
from constants import CLUSTER_SCHEMA
import os
import copy


counter = 0
for cluster_id in range(2):
    cluster_i_size = np.random.randint(1, 5)
    cluster_i = []
    cluster_item = [_ for _ in range(len(CLUSTER_SCHEMA))]
    
    for item_id in range(cluster_i_size):
        example_key = f'datafolder/example_{round(np.random.rand(), 2)}'
        
        cluster_item[CLUSTER_SCHEMA["example_key"]["id"]] = example_key
        cluster_item[CLUSTER_SCHEMA["example_emb_id"]["id"]] = counter
        cluster_item[CLUSTER_SCHEMA["distance_to_centroid"]["id"]] = round(1 - np.random.uniform(-1, 1), 2)
        cluster_item[CLUSTER_SCHEMA["cluster_id"]["id"]] =  cluster_id
        
        counter += 1
        
        cluster_i.append(copy.deepcopy(cluster_item))  
        
    sort_descending = True
    cluster_i_sorted = sorted(
            cluster_i,
            key=lambda x: float(x[CLUSTER_SCHEMA["distance_to_centroid"]["id"]]),
            reverse=sort_descending,
        )  

    os.makedirs(f"test_data/sorted_clusters", exist_ok=True)
    np.save(f"test_data/sorted_clusters/cluster_{cluster_id}.npy", cluster_i_sorted)
        
        