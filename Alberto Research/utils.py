import spacy
import numpy as np
import torch

def split_markdown_to_paras(text, spacy_model="fr_core_news_sm", n_sents_per_para=10):
    
    nlp = spacy.load(spacy_model)
    doc = nlp(text)
    sents = [sent.text for sent in doc.sents]
    
    rang_sentence_union = np.arange(start=0, stop=len(sents), step=n_sents_per_para)
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

def compute_norm_embeddings(tokenizer, model, sentence):

    tokenized_sentences = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model(**tokenized_sentences)[0][:, 0].squeeze(0) # to take out the unused dimension since we are not batching
        # print(embeddings.shape)

    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)
    detached_embeddings = normalized_embeddings.detach().cpu().numpy() # detached into cpu so that we can manipulate them for clustering

    torch.cuda.empty_cache() # careful with running out of memory

    return detached_embeddings

def cluster_n(cluster_model, n_clusters, embeddings, scoring_function):

    clusters = cluster_model.fit_predict(embeddings)
    sil_sc = scoring_function(embeddings, clusters)

    print("Number of clusters: ", n_clusters)
    print("Score: ", sil_sc)
    print()

    return clusters, sil_sc