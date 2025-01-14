from llama_parse import LlamaParse
import pandas as pd
import urllib.request
import uuid
import nest_asyncio
import time



def csv_add_markdown_col(data, llama_parse_key): 
    
    nest_asyncio.apply()
    parser = LlamaParse(
        # can also be set in your env as LLAMA_CLOUD_API_KEY
        api_key=llama_parse_key, # Put your own API key
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="fr",  # Optionally you can define a language, default=en
        disable_ocr=False # This is the default option
    )

    print(len(data.index.tolist()))
    md_column = []
    starting_time = time.time()
    pdfs = []
    ids = []
    for index, row in data.iterrows():
        doc_id = row["doc_id"]
        cache_url = row["cache"]
        uuid_ = uuid.uuid1()
        pdf_path = "pdfs/" + str(uuid_)+".pdf"
        print(doc_id)
        print(cache_url)
        
        try:
            print("error here")
            urllib.request.urlretrieve(cache_url, pdf_path)
            document = parser.load_data(pdf_path)
            md = ""
            for line in document:
                md += line.text
                md += "\n\n"
            md_column.append (md)
            
        except :
            print("Couldn't download")
            md_column.append ("Download failed")
            
    data["markdown"] = md_column
    print(time.time() - starting_time)

    return data