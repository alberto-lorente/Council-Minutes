from council_rag.preprocessing import preprocess_markdown_text
from council_rag.data_transformations import process_tables, summarize_clusters
from council_rag.rag import prepare_data_for_rag, set_up_rag_index

def main(markdown, 
        embd_model, 
        spacy_model, 
        sent_per_para, 
        device, 
        pdf_path, 
        table_processing_prompt, 
        groq_token, 
        summary_prompt, 
        model, 
        token_limit, 
        sleep_time,
        processed_tables_dict, 
        splitter,
        chunk_size,
        chunk_overlap,
        length_function,
        is_separator_regex,
        embedding_model_rag):

    paragraphs_list, clusters_dict, model = preprocess_markdown_text(markdown=markdown,
                                                        model_id =embd_model, 
                                                        spacy_model=spacy_model, 
                                                        n_sents_per_para=sent_per_para,
                                                        device=device)

    processed_tables = process_tables(pdf_path, 
                                table_processing_prompt, 
                                groq_token)

    clusters_dict = summarize_clusters(clusters_dict, 
                                    summary_prompt, 
                                    groq_token, 
                                    model=model, 
                                    token_limit=token_limit, 
                                    sleep_time=sleep_time)
    
    clusters_list, paragraphs_list, cluster_summ_docs, cluster_chunks, augmented_table_chunks, table_descriptions_chunks, html_tables_chunks = prepare_data_for_rag(clusters_dict, 
                                                                                                                                                                    paragraphs_list, 
                                                                                                                                                                    processed_tables_dict, 
                                                                                                                                                                    splitter,
                                                                                                                                                                    chunk_size,
                                                                                                                                                                    chunk_overlap,
                                                                                                                                                                    length_function,
                                                                                                                                                                    is_separator_regex)
    vector_store, index, emd_dims = set_up_rag_index(embedding_model_rag)
    
    
    
    
    
    
    
    return 

if __name__ == "__main__":
    main()