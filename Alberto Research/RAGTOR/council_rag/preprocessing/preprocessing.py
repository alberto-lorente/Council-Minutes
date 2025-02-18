import torch
from transformers import AutoTokenizer, AutoModel
import spacy
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import gc

device = "cpu"
if torch.cuda.is_available():
    # print("Cuda available")
    device = torch.device('cuda')

def split_markdown_to_paras(text, spacy_model="fr_core_news_sm", n_sents_per_para=8):
    """
    Splits the markdown text into paragraphs based on the number of sentences per paragraph with spacy.
    Returns the list of paragraphs dictionaries with the total text of the paragraph, the start and stop ranges of the sentences and the list of sentences.
    """
    
    nlp = spacy.load(spacy_model)
    doc = nlp(text)
    sents = [sent.text for sent in doc.sents]
    
    stop = len(sents)
    rang_sentence_union = np.arange(start=0, stop=stop, step=n_sents_per_para)
    # print("Total number of sents: ", len(sents))
    # print("Number of final chunks: ", len(rang_sentence_union))


    paragraphs = []
    i = 0

    # merging sentences based on the ranges
    while i+1 < len(rang_sentence_union):
        start = rang_sentence_union[i]
        stop = rang_sentence_union[i+1]
        # print(start, stop)

        subset_to_join = sents[start : stop]
        sent_union = " ".join(subset_to_join)

        paragraph_info = {"paragraph_union": sent_union,
                        "start_range": start,
                        "stop_range": stop,
                        "list_sents":subset_to_join}

        paragraphs.append(paragraph_info)

        i += 1

    # if the stop of the range comes before the last sentence, we take those final couple of sentences
    # and add them to the last sentence of the paragraph list
    if stop != len(sents):

        subset_to_join = sents[stop : len(sents)]
        final_sents = " ".join(subset_to_join)
        para_to_edit = paragraphs.pop(-1)
        final_union = para_to_edit["paragraph_union"] + " " + final_sents

        para_to_edit["paragraph_union"] =  final_union
        para_to_edit["stop_range"] = len(sents)
        para_to_edit["list_sents"].extend(subset_to_join)
        paragraphs.append(para_to_edit)
        
    return paragraphs

# not used anymore
# def compute_norm_embeddings(tokenizer, model, sentence, device="cuda"):

#     tokenized_sentences = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)

#     model.eval()
#     with torch.no_grad():
#         embeddings = model(**tokenized_sentences)[0][:, 0].squeeze(0) # to take out the unused dimension since we are not batching
#         # print(embeddings.shape)

#     normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)
#     detached_embeddings = normalized_embeddings.detach().cpu().numpy() # detached into cpu so that we can manipulate them for clustering

#     torch.cuda.empty_cache() # careful with running out of memory

#     return detached_embeddings

# def compute_norm_embeddings(model_id, sentence, device="cuda"):

#     model = SentenceTransformer(model_id)
#     embeddings = model.encode(sentence, trust_remote_code=True)

#     # detached_embeddings = embeddings.detach().cpu().numpy() # detached into cpu so that we can manipulate them for clustering

#     torch.cuda.empty_cache() # careful with running out of memory

#     return embeddings

def unload_cuda():
    """
    Function to workaround CUDA issues.
    """
    torch.cuda.empty_cache()
    
    
    return gc.collect()

def compute_norm_embeddings(model, sentence):
    """
    Computes the embeddings of the sentence using the model.
    The model is passed already initialized to avoid having to load it again and running into CUDA issues.
    Returns the embeddings.
    """
    embeddings = model.encode(sentence)
    unload_cuda()
    
    return embeddings

def compute_paragraph_embeddings(paragraphs, model):
    """
    Computes the embeddings of the paragraphs using the model.
    The model is passed already initialized to avoid having to load it again and running into CUDA issues.
    Returns the paragraphs list with the embeddings as an item of each paragraph dictionary.
    """
    # Compute the embeddings for each paragraph
    list_paras = [para["paragraph_union"] for para in paragraphs]
    for paras, para_dict in zip(list_paras, paragraphs):
        para_dict["para_embedding"] = compute_norm_embeddings(model, paras)
    return paragraphs

def cluster_n(cluster_model, n_clusters, embeddings, scoring_function):
    """
    Clusters the embeddings using the cluster model, a Gaussian Mixture.
    Returns the clusters and the silhouette score.
    """
    clusters = cluster_model.fit_predict(embeddings)
    sil_sc = scoring_function(embeddings, clusters)

    # print("Number of clusters: ", n_clusters)
    # print("Score: ", sil_sc)
    # print()

    return clusters, sil_sc

def get_optimal_n_clusters(squeezeded_embeddings, min_n_clusters, max_n_clusters=12):
    """
    Gets the optimal number of clusters for the embeddings based on the max silhouette score.
    The range of clusters to test is from 8 to max_n_clusters. Ideally it would be computed dinamycally based on the data.
    Returns the optimal number of clusters, the final clusters labels for the optimal number of clusters and the silhouette scores.
    """
    
    #ranges of clusters to test
    range_clusters = np.arange(start=min_n_clusters, stop=max_n_clusters, step=1)

    # Compute the silhouette scores for each number of clusters
    silhouette_scores = []
    clusters_labels = []
    for n_cluster in range_clusters:
        gm = GaussianMixture(n_components=n_cluster, random_state=42) # using Gaussian Mixture as in the paper references
        clusters, sil_sc = cluster_n(gm, n_cluster, squeezeded_embeddings, silhouette_score)
        silhouette_scores.append(sil_sc)
        clusters_labels.append(clusters) # saving the labels so that we don't need to recompute them after getting the optimal n

    # Getting the optimal number of clusters
    max = np.argmax(silhouette_scores)
    optimal_n = range_clusters[max]
    # print("Index", max)
    # print("Optimal Number of Clusters", optimal_n)

    # Getting the labels for the optimal number of clusters
    final_clusters = clusters_labels[max]
    
    return optimal_n, final_clusters, silhouette_scores


# Filling the dictionary of clusters
def fill_clusters_dict(paragraphs, clusters_ids, final_clusters, recompute_embeddings=False, model=None):
    """
    Processes prev. data into a the dictionary of clusters with the clusters labels and text for each cluster and the indexes of the clustered paragraphs.
    Returns the clusters dictionary and the paragraphs list.
    """
    i = 0

    for para_dict, cluster in zip(paragraphs, final_clusters):
        cluster_n_string = f"cluster_{cluster}"
        para_dict["para_cluster"] = cluster_n_string
        clusters_ids[cluster_n_string]["para_indexes"].append(i)
        clusters_ids[cluster_n_string]["union_paras"] = clusters_ids[cluster_n_string]["union_paras"] + para_dict["paragraph_union"]
        i = i + 1

    # Compute the embeddings for each cluster
    if recompute_embeddings:
        for cluster in clusters_ids.keys():
            text = clusters_ids[cluster]["union_paras"]
            cluster_embds = compute_norm_embeddings(model, text)
            clusters_ids[cluster]["cluster_embedding"] = cluster_embds

    return clusters_ids, paragraphs
