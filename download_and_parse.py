from llama_parse import LlamaParse
import pandas as pd
import urllib.request
import uuid
import nest_asyncio
import time


nest_asyncio.apply()


parser = LlamaParse(
    # can also be set in your env as LLAMA_CLOUD_API_KEY
    api_key="llx-...", # Put your own API key
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",  # Optionally you can define a language, default=en
    disable_ocr=False # This is the default option
)

csv = pd.read_csv("dataset.csv", sep=",")
csv = csv.sample(frac=1)
csv = csv.head(10)
print(len(csv.index.tolist()))
md_column = []
starting_time = time.time()
pdfs = []
ids = []
for index, row in csv.iterrows():
    doc_id = row["doc_id"]
    cache_url = row["cache"]
    uuid_ = uuid.uuid1()
    pdf_path = "pdfs/" + str(uuid_)+".pdf"
    print(doc_id)
    print(cache_url)
    try:
        urllib.request.urlretrieve(cache_url, pdf_path)
        document = parser.load_data(pdf_path)
        md = ""
        for line in document:
            md += line.text
            md += "\n\n"
        md_column.append(md)
    except :
        print("Couldn't download")
        md_column.append ("Download failed")
csv["markdown"] = md_column
csv.to_csv("dataset_with_md.csv")
print(time.time() - starting_time)
