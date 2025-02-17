import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from .preprocessing import compute_paragraph_embeddings, split_markdown_to_paras, get_optimal_n_clusters, fill_clusters_dict
from sentence_transformers import SentenceTransformer

device = "cpu"
if torch.cuda.is_available():
    # print("Cuda available")
    device = torch.device('cuda')


def preprocess_markdown_text(markdown,
                            model_id ="Jaume/gemma-2b-embeddings", 
                            spacy_model="fr_core_news_sm", 
                            n_sents_per_para=10, 
                            device=device):

        
    # Split the markdown into paragraphs, markdown is a string
    paragraphs = split_markdown_to_paras(markdown, spacy_model, n_sents_per_para)


    model = SentenceTransformer(model_id)
    # Split the markdown into paragraphs
    paragraphs = compute_paragraph_embeddings(paragraphs, model)
    
    # Get just the embeddings to compute the ideal number of clusters
    squeezeded_embeddings = [para_dict["para_embedding"] for para_dict in paragraphs]

    # Compute the ideal number of clusters
    optimal_n, final_clusters, silhouette_scores = get_optimal_n_clusters(squeezeded_embeddings, max_n_clusters=9)
    
    # Creating the dictionary of clusters
    clusters_ids = {f"cluster_{cluster_id}": {"para_indexes": [],
                                            "union_paras": ""} for cluster_id in np.arange(0, optimal_n, 1)}


    # Compute the embeddings for each cluster
    clusters, paragraphs = fill_clusters_dict(paragraphs, 
                                                clusters_ids, 
                                                final_clusters, 
                                                recompute_embeddings=False, 
                                                model=None)
    return paragraphs, clusters, model #so that we don't have to re-load it the model
