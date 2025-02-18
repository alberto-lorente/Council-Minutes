
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..data_transformations.text_transformations import generate_groq_summary
from ..preprocessing.preprocessing import unload_cuda

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
    shape_emb = embedding_model.encode("Hello World!")
    emd_dims =  shape_emb.shape[0]
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
        embed = model.encode(doc.page_content)
        all_embeddings.append(embed)
        
    metadatas = [doc.metadata for doc in docs]
    page_contents = [doc.page_content for doc in docs]
    content_emb_tupe = tuple(zip(page_contents, all_embeddings))
    vector_store = vector_store.add_embeddings(content_emb_tupe, metadatas)
    
    return vector_store, all_embeddings
    

