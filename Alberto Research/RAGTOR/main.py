from council_rag.preprocessing import preprocess_markdown_text
from council_rag.data_transformations import process_tables, summarize_clusters

def main(markdown, 
        embd_model, 
        spacy_model, 
        sent_per_para, 
        device, 
        pdf_path, 
        base_prompt, 
        groq_token, 
        summary_prompt, 
        model, 
        token_limit, 
        sleep_time):

    paragraphs_dict, clusters_dict = preprocess_markdown_text(markdown=markdown,
                                                        model_id =embd_model, 
                                                        spacy_model=spacy_model, 
                                                        n_sents_per_para=sent_per_para,
                                                        device=device)

    processed_tables = process_tables(pdf_path, 
                                base_prompt, 
                                groq_token)

    clusters_dict = summarize_clusters(clusters_dict, 
                                    summary_prompt, 
                                    groq_token, 
                                    model=model, 
                                    token_limit=token_limit, 
                                    sleep_time=sleep_time)
    
    return paragraphs_dict, clusters_dict, processed_tables

if __name__ == "__main__":
    main()