# - For each document type, create two clusters.

import pandas as pd
import numpy as np
import pickle
import json
import ast

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import cdist


def create_evaluate_cluster(clustering_algorithm, X, n_clusters, nearest_centroid=False):
    
    clustering_algorithm.fit(X)
    labels = clustering_algorithm.labels_
    
    sil_score = silhouette_score(X, labels, random_state=42)
    cal_score = calinski_harabasz_score(X, labels)

    if nearest_centroid == True:
        cluster_centers = NearestCentroid().fit(X, labels).centroids_
    else:
        cluster_centers = clustering_algorithm.cluster_centers_
        
    distorsion = sum(np.min(cdist(X, cluster_centers, 'euclidean'), axis=1)) / len(X)
    
    return clustering_algorithm, sil_score, cal_score, distorsion, labels


data_path = "../../dataset_full_texts.csv"
data = pd.read_csv(data_path)

print("Data Loaded")

data_document_types = set(data["nature"].to_list())
print(f"Document types found:{data_document_types}")


def reformat_embeddings(embbeding):
    
    embedding_clean_string = embbeding[1:-1].replace("\n", "")
    embedding_clean_elements = embedding_clean_string.strip(" ").split(" ")
    embedding_float_type = [float(numb) for numb in embedding_clean_elements if not " " in numb and numb != '']
    
    # print(type(embedding_float_type))
    
    return embedding_float_type

data["typed_embeddings"] = data["embeddings"].apply(reformat_embeddings)

cluster_logs = {}

for document_type in data_document_types:
    
    cluster_logs[document_type] = {}
    
    print(f"Working on {document_type} type.")
    
    data_doc_filter = data[data["nature"] == document_type]
    
    if len(data_doc_filter) < 4:
        continue
    
    data_doc_embs = data_doc_filter["typed_embeddings"].to_list()    
    
    kmeans_model, sil_score, cal_score, distorsion, labels = create_evaluate_cluster(KMeans(n_clusters=2, random_state=40), data_doc_embs, 2)
    data_doc_filter["km_cluster_labels"] = labels

    cluster_logs[document_type]["doc_type"] = document_type
    cluster_logs[document_type]["sil_score"] = sil_score
    cluster_logs[document_type]["cal_score"] = cal_score
    cluster_logs[document_type]["distorsion"] = distorsion
    
    print("Clustering for doc type done.\n")
    print(cluster_logs[document_type])

    pickle_cluster_algo_name = f"kmeans_{str(document_type).replace(".", "_")}.pkl"
    pickle.dump(kmeans_model, open(pickle_cluster_algo_name, "wb"))
        
    print("Clustering model saved.\n")
        
    data_annotated_doc_type_name = f"data_{str(document_type).replace(".", "_")}_annotated.csv"
    data_doc_filter.to_csv(data_annotated_doc_type_name)
    print("Annotated data saved")

with open("clusters_info.json", "w") as f:
    json.dump(cluster_logs, f)