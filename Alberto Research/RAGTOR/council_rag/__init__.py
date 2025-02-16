import torch
from transformers import AutoTokenizer, AutoModel
import spacy
from huggingface_hub import HfFolder, whoami

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from .preprocessing import compute_paragraph_embeddings, split_markdown_to_paras, get_optimal_n_clusters, fill_clusters_dict

def preprocess_markdown_text(markdown,
                            hf_token_path,
                            model_id ="HIT-TMG/KaLM-embedding-multilingual-mini-v1", 
                            spacy_model="fr_core_news_sm", 
                            n_sents_per_para=10):

    device = "cpu"
    if torch.cuda.is_available():
        print("Cuda available")
        device = torch.device('cuda')
        
    with open(hf_token_path, "r") as f:
        hf_token = f.read()
    HfFolder.save_token(hf_token)
    print(whoami()["name"])
        
    # Split the markdown into paragraphs, markdown is a string
    paragraphs = split_markdown_to_paras(markdown, spacy_model, n_sents_per_para)

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    # Split the markdown into paragraphs
    paragraphs = compute_paragraph_embeddings(paragraphs, tokenizer, model)
    
    # Get just the embeddings to compute the ideal number of clusters
    squeezeded_embeddings = [para_dict["para_embedding"] for para_dict in paragraphs]

    # Compute the ideal number of clusters
    optimal_n, final_clusters = get_optimal_n_clusters(squeezeded_embeddings, max_n_clusters=9)
    
    # Creating the dictionary of clusters
    clusters_ids = {f"cluster_{cluster_id}": {"para_indexes": [],
                                            "union_paras": ""} for cluster_id in np.arange(0, optimal_n, 1)}


    # Compute the embeddings for each cluster
    clusters_ids, paragraphs = fill_clusters_dict(paragraphs, 
                                                clusters_ids, 
                                                final_clusters, 
                                                recompute_embeddings=False, 
                                                tokenizer=None, 
                                                model=None)
    return paragraphs, clusters_ids
