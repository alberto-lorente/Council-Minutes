import torch
from huggingface_hub import HfFolder, whoami
from table_transformations import format_table_info, extract_tables_from_page, convert_pdf_to_image, augment_multimodal_context
from text_transformations import generate_groq_summary
import time


with open("HF_TOKEN.txt", "r") as f:
    hf_token = f.read()

with open("GROQ_KEY.txt", "r") as f:
    groq_token = f.read()

HfFolder.save_token(hf_token)
# print(whoami()["name"])

device = "cpu"
if torch.cuda.is_available():
    # print("Cuda available")
    device = torch.device('cuda')
    

def process_tables(pdf_path, base_prompt, groq_token):

    extracted_tables = extract_tables_from_page(pdf_path)
    extracted_tables_filter = {k:v for k, v in extracted_tables.items() if v != []} # filtering the pages that actually have tables
    list_pages_with_tables = list(extracted_tables_filter.keys())
    
    # converting the pdf pages into images
    image_paths = convert_pdf_to_image(pdf_path)  
    table_descriptions = [augment_multimodal_context(im_path, base_prompt, groq_token) for im_path in image_paths]
    
    # getting table htmls
    list_table_objects = list(extracted_tables_filter.values())
    list_table_objects = [table[0] for table in list_table_objects] # unfolding the list of lists
    list_table_htmls = [table.html for table in list_table_objects]
    
    # processing them into a manageable format
    processed_tables = format_table_info(list_pages_with_tables, list_table_htmls, table_descriptions)
    
    return processed_tables

def summarize_clusters(cluster_paras, summary_prompt, groq_token, model="gemma2-9b-it", token_limit=14000, sleep_time=60):

    token_count = 0
    for cluster_para in cluster_paras:
        # print(len(cluster_para))
        naive_length_check = len(cluster_para.split(" ")) # the token calcs in groq are very close to this number
        token_count += naive_length_check
        # print(naive_length_check)
        if token_count > token_limit:
            time.sleep(sleep_time)
            token_count = 0
        cluster_summary = generate_groq_summary(summary_prompt, cluster_para, groq_token, model)
        
        cluster_para["cluster_summary"] = cluster_summary
        
    return cluster_paras