from llama_parse import LlamaParse
import nest_asyncio
import os

def dir_of_downloaded_pdf_to_markdown(data_folder, llama_parse_key): 
    
    """
    Input a folder with downloaded pdfs and an Llama Parse API Key and:
    1. Turn the PDF to markdown.
    2. Save the markdown version in a new .md file with the same name as the PDF.
    3. Returns a list of the markdown versions of the PDFs.
    """
    
    nest_asyncio.apply()
    
    parser = LlamaParse(
        # can also be set in your env as LLAMA_CLOUD_API_KEY
        api_key=llama_parse_key,
        result_type="markdown", 
        num_workers=4,  # could be passed to a list of pdf paths
        verbose=True,
        language="fr",  
        disable_ocr=False # This is the default option
    )
    
    list_pdfs_files = os.listdir(data_folder)
    
    markdowns = []
    
    for pdf_file in list_pdfs_files:
        
        pdf_file_path = os.path.join(data_folder, pdf_file)
        markdown = parser.load_data(pdf_file_path)
        
        with open(f'{pdf_file_path.strip(".pdf")}_markdown.md', 'w') as f:
            for doc in markdown:
                f.write(doc.text + '\n')
                
            markdowns.append(f)
            
    return markdowns