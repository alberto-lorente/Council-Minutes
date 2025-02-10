import torch
from transformers import AutoTokenizer, AutoModel
import spacy
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from utils import split_markdown_to_paras, compute_norm_embeddings, cluster_n

with open("HF_TOKEN.txt", "r") as f:
    hf_token = f.read()

device = "cpu"
if torch.cuda.is_available():
    print("Cuda available")
    device = torch.device('cuda')
    
model_id = "HIT-TMG/KaLM-embedding-multilingual-mini-v1" # which model to use?
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
model = AutoModel.from_pretrained(model_id).to(device)

markdown_path = input("Input valid markdown file.")
with open(markdown_path, "r") as f:
    markdown = f.read()

paragraphs = split_markdown_to_paras(markdown)
list_paras = [para["paragraph_union"] for para in paragraphs]
for paras, para_dict in zip(list_paras, paragraphs):
    para_dict["para_embedding"] = compute_norm_embeddings(tokenizer, model, paras)
        
squeezeded_embeddings = [para_dict["para_embedding"] for para_dict in paragraphs]

range_clusters = np.arange(start=3, stop=9, step=1)

silhouette_scores = []
clusters_labels = []
for n_cluster in range_clusters:
    gm = GaussianMixture(n_components=n_cluster, random_state=42)
    clusters, sil_sc = cluster_n(gm, n_cluster, squeezeded_embeddings, silhouette_score)
    silhouette_scores.append(sil_sc)
    clusters_labels.append(clusters)

max = np.argmax(silhouette_scores)
optimal_n = range_clusters[max]
print("Index", max)
print("Optimal Number of Clusters", optimal_n)
final_clusters = clusters_labels[max]

clusters_ids = {f"cluster_{cluster_id}": {"para_indexes": [],
                                        "union_paras": ""} for cluster_id in np.arange(0, optimal_n, 1)}

i = 0

for para_dict, cluster in zip(paragraphs, final_clusters):
    cluster_n_string = f"cluster_{cluster}"
    para_dict["para_cluster"] = cluster_n_string
    clusters_ids[cluster_n_string]["para_indexes"].append(i)
    clusters_ids[cluster_n_string]["union_paras"] = clusters_ids[cluster_n_string]["union_paras"] + para_dict["paragraph_union"]
    i = i + 1
    
for cluster in clusters_ids.keys():
    text = clusters_ids[cluster]["union_paras"]
    cluster_embds = compute_norm_embeddings(tokenizer, model, text)
    clusters_ids[cluster]["cluster_embedding"] = cluster_embds