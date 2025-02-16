
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    
    clusters_list, paragraphs_list = preprocess_dicts_for_rag(clusters_dict, paragraphs_list)
    
    
    cluster_summ_docs, cluster_chunks = process_cluster_dict_to_objects(clusters_list, 
                                                                        splitter=splitter,
                                                                        chunk_size=chunk_size,
                                                                        chunk_overlap=chunk_overlap,
                                                                        length_function=length_function,
                                                                        is_separator_regex=is_separator_regex)
    
    augmented_table_chunks, table_descriptions_chunks, html_tables_chunks = process_tables_dict_to_objects(processed_tables_dict)
    
    return clusters_list, paragraphs_list, cluster_summ_docs, cluster_chunks, augmented_table_chunks, table_descriptions_chunks, html_tables_chunks
    
    
    
    

