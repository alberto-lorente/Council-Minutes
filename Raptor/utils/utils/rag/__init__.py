
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..data_transformations.text_transformations import generate_groq_summary
from ..preprocessing.preprocessing import unload_cuda
from groq import Groq

from huggingface_hub import HfFolder, whoami

with open("example_md_to_text.txt", "r", encoding="latin-1") as f: # TO DO: check proper encoding for .md files
    markdown_example = f.read()

with open("example.pdf", "rb") as f:
    pdf = f.read()

with open("HF_TOKEN.txt", "r") as f:
    hf_token = f.read()

with open("GROQ_KEY.txt", "r") as f:
    groq_token = f.read()
    
HfFolder.save_token(hf_token)

device = "cpu"
if torch.cuda.is_available():
    print("Cuda available")
    device = torch.device('cuda')

def preprocess_dicts_for_rag(cluster_dict, paragraphs_list):
    """
    Lightly preprocesses the dictionary of clusters and the list of paragraphs for the RAG.
    Returns the list of clusters and the list of paragraphs.
    """
    clusters_list = list(cluster_dict.values())
    i = 0
    while i < len(clusters_list):
        clusters_list[i]["cluster"] = i
        i += 1
    try:
        del paragraphs_list[0]["para_embedding"]
    except:
        pass
    return clusters_list, paragraphs_list

def set_up_rag_index(embedding_model):
    """
    Sets up a Faiss RAG index based on the dimensions of the embedding model.
    Returns the vector store, the index and the embedding dimensions.
    """
    shape_emb = embedding_model.embed_documents(["Hello World!"])
    emd_dims =  len(shape_emb[0]) # huf in langchain returns a list of shape num_docs[emb_dims]
    index = faiss.IndexFlatL2(emd_dims)
    vector_store = FAISS(embedding_model, 
                    index, 
                    InMemoryDocstore({}), 
                    {})
    
    return vector_store, index, emd_dims

def process_cluster_dict_to_objects(clusters_list, 
                                splitter=RecursiveCharacterTextSplitter,
                                chunk_size=450,
                                chunk_overlap=35,
                                length_function=len,
                                is_separator_regex=False):
    """
    Processes the cluster dictionary into a list of Langchain Document objects.
    Processes one list for the cluster summaries and one for the individual chunks.
    Returns the two list of Document objects.
    """
    cluster_paras = [] # to process into smaller chunks
    cluster_summ_docs = [] # filled with the cluster summaries langchain doc type

    for cluster in clusters_list:
        
        cluster_para = cluster["union_paras"]
        cluster_summ = cluster["cluster_summary"]
        
        cluster_paras.append(cluster_para)
        cluster_summ_docs.append(Document(page_content=cluster_summ, metadata={"cluster": cluster["cluster"],
                                                                                "type": "summary"}))
        
    splitter = splitter(chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap,
                        length_function=length_function,
                        is_separator_regex=is_separator_regex)
        
    cluster_paras_docs = []
    for cluster_union in cluster_paras:
        cluster_union_docs = splitter.split_text(cluster_union)
        cluster_paras_docs.append(cluster_union_docs)
        
    cluster_chunks = []
    i = 0
    while i < len(cluster_paras_docs):  
        curr_cluster = cluster_paras_docs[i]
        for cluster_chunk in curr_cluster:
            cluster_chunks.append(Document(page_content=cluster_chunk, metadata={"cluster": i,
                                                                            "type": "cluster_chunk"}))
        i += 1
        
    return cluster_summ_docs, cluster_chunks
    
def process_tables_dict_to_objects(processed_tables_dict):
    """
    Processes the tables dictionary into three list of Langchain Document objects: one for the augmented table chunks, one for the table descriptions and one for the html tables.
    Returns the list of augmented table chunks, the list of table descriptions and the list of html tables.
    """
    augmented_table_chunks = []
    table_descriptions_chunks = []
    html_tables_chunks = []

    for table in processed_tables_dict:
        
        augmented_chunk = table["table_augmented_context"]
        augmented_table_chunks.append(Document(page_content=augmented_chunk, metadata={"type": "augmented_table"}))
        
        description_chunk = table["table_context"]
        table_descriptions_chunks.append(Document(page_content=description_chunk, metadata={"type": "description_table"}))
        
        html_chunk = table["table_html"]
        html_tables_chunks.append(Document(page_content=html_chunk, metadata={"type": "html_table"}))
        
    return augmented_table_chunks, table_descriptions_chunks, html_tables_chunks

        
def prepare_data_for_rag(clusters_dict, 
                        paragraphs_list, 
                        processed_tables_dict, 
                        splitter,
                        chunk_size,
                        chunk_overlap,
                        length_function,
                        is_separator_regex):
    """
    Prepares the data for the RAG.
    Returns the list of clusters, the list of paragraphs and the list of all documents processed.
    """
    clusters_list, paragraphs_list = preprocess_dicts_for_rag(clusters_dict, paragraphs_list)
    
    
    cluster_summ_docs, cluster_chunks = process_cluster_dict_to_objects(clusters_list, 
                                                                        splitter=splitter,
                                                                        chunk_size=chunk_size,
                                                                        chunk_overlap=chunk_overlap,
                                                                        length_function=length_function,
                                                                        is_separator_regex=is_separator_regex)
    
    augmented_table_chunks, table_descriptions_chunks, html_tables_chunks = process_tables_dict_to_objects(processed_tables_dict)
    all_docs = cluster_chunks + cluster_summ_docs + augmented_table_chunks + table_descriptions_chunks + html_tables_chunks
    
    return clusters_list, paragraphs_list, all_docs

def shorten_summary_docs(all_docs, groq_token, model="gemma2-9b-it"):
    """
    Prompts Gemma to shortens the documents whose context exceeds 1500 naive tokens.
    Returns the list of Document objects with the shortened summaries.
    """
    summary_shorten_prompt="Résumez le texte suivant en moins de 1400 mots: {}. Output le résumé directement."
    for doc in all_docs:
        if len(doc.page_content.split(" ")) > 1500:
            new_content = generate_groq_summary(summary_shorten_prompt, doc.page_content, groq_token, model)
            doc.page_content = new_content
    
    return all_docs

    
def populate_vector_store(vector_store, docs, model):
    """
    Calculates embeddings for the docs and populates the vector store with the embeddings and the documents.
    Returns the vector store and the list of embeddings.
    """
    all_embeddings = []
    for doc in docs:
        unload_cuda()
        embed = model.embed_documents([doc.page_content])
        all_embeddings.append(embed[0])
        
    metadatas = [doc.metadata for doc in docs]
    page_contents = [doc.page_content for doc in docs]
    content_emb_tupe = tuple(zip(page_contents, all_embeddings))
    vector_store.add_embeddings(content_emb_tupe, metadatas)
    
    return vector_store, all_embeddings    

def raptor_query_vector_store(vector_store, query):
        """
        Performs the RAPTOR query on the vector store.
        Returns the list of relevant facts and a relevant tables.
        """
        results_query_level_one = vector_store.similarity_search(
                                                                query,
                                                                k=1,
                                                                filter={"type": "summary"}, 
                                                                )
        unload_cuda()

        cluster_level_one = results_query_level_one[0].metadata["cluster"]

        # now we search only in the cluster retrieved in the first step
        results_query_level_two = vector_store.similarity_search(
                                                        query,
                                                        k=5,
                                                        filter={"cluster": cluster_level_one,
                                                                "type": "cluster_chunk"}, 
                                                        )
        unload_cuda()
        results_query_tables = vector_store.similarity_search(
                                                        query,
                                                        k=1,
                                                        filter={"type": 'description_table'}, # {"$in": []} 
                                                        )
        unload_cuda()

        relevant_facts = [sent_query_level_two.page_content for sent_query_level_two in results_query_level_two]
        relevant_table = results_query_tables[0].page_content
        unload_cuda()

        return relevant_facts, relevant_table

def augment_query_rag(query, relevant_facts, relevant_table):
    """
    Augments the original query with the relevant facts and the relevant table.
    We assume that the original query does not have a format string.
    Returns the augmented query.
    """
    aug_prompt="""\nVoici quelques faits pertinents pour vous aider à répondre et una description du qui peut s'avérer utile. 
    Si la description du n'est pas utile, ignorez-le.\n"""
    
    augmented_data_string = aug_prompt + "\n".join(relevant_facts) + "\n" + relevant_table
    
    # formated_augmented_query = query.format(augmented_data_string) # the original query has to have a format string
    formated_augmented_query = query + augmented_data_string       # the original query does not have a format string
    
    return formated_augmented_query


def raptor_query_all_prompts(vector_store, prompts_query):
    """
    Uses the raptor_query_vector_store function to retrieve the relevant facts and the relevant table for each prompt in the prompts_query dictionary.
    Returns a list of dictionaries with the prompt_id, the formated augmented query, the relevant facts and the relevant table.
    """
    list_aug_queries = []
    for k, v in prompts_query.items():
        relevant_facts, relevant_table = raptor_query_vector_store(vector_store, v)
        formated_augmented_query = augment_query_rag(v, relevant_facts, relevant_table)
        dict_query = {"prompt": k,
                        "formated_augmented_query": formated_augmented_query,
                        "facts": relevant_facts,
                        "table": relevant_table}
        list_aug_queries.append(dict_query)
    return list_aug_queries

def final_query(query, groq_key):
    """
    Performs the final query on the model.
    Returns the response of the model.
    """
    client = Groq(api_key=groq_key)
    
    messages = [
                {"role": "system",
                "content":  "Vous êtes un assistant utile" },
                {"role": "user",
                "content":  query}
                ]
    
    chat_completion = client.chat.completions.create(messages=messages, model="gemma2-9b-it")
    return chat_completion.choices[0].message.content

def return_final_responses(list_aug_queries, groq_key):
    
    for query_dict in list_aug_queries:
        query = query_dict["formated_augmented_query"]
        response = final_query(query, groq_key)
        query_dict["final_response"] = response
    
    return list_aug_queries