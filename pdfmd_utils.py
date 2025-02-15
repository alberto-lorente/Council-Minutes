import pandas as pd
import urllib.request
import uuid
import nest_asyncio
import time
import fitz



nest_asyncio.apply()

from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-XrrnoH7Vou2A7NoeIhrXU1Ih6WyDLo3GOKt3v8VuxzgvbxKr", 
    result_type="markdown", 
    num_workers=4, 
    verbose=True,
    language="en", 
    disable_ocr = True
)

def pdf2md (file_path):
	"""
	Converts a locally stored pdf file to markdown format 
	"""

	document = parser.load_data(file_path)
	md = ""
	for line in document:
		md += line.text
		md += "\n\n"
	return md

def download_pdf (url, pdf_path):
	""" Downloads the pdf at url and store in pdf_path. return True if success, else False
	"""
	try:
		urllib.request.urlretrieve(url, pdf_path)
		return True
	except:
		print("Could not download", url)
		return False



def process_row (row, pagebypage=False):
	""" Process a line of the dataset """
	doc_id = row["doc_id"]
	cache_url = row["cache"]
	uuid_ = uuid.uuid1 ()
	pdf_path = "pdfs/" + str(uuid_)+".pdf"
	if download_pdf(cache_url, pdf_path):
		if pagebypage:
    			return pdf2mdpagebypage(pdf_path)
        	else:
            		return pdf2md (pdf_path)
        
	else:
		print("Could not process", doc_id)
		return None


def pf2mdpagebypage(input_pdf, output_folder):
    doc = fitz.open(input_pdf)
    pages = []
    for i, page in enumerate(doc):
        new_pdf = fitz.open()
        new_pdf.insert_pdf(doc, from_page=i, to_page=i)
        
        output_filename = f"{output_folder}/page_{i+1}.pdf"
        new_pdf.save(output_filename)
        new_pdf.close()
        pages.append (pdf2md(output_filename))
    return pages



# Example use !
if __name__ == "main":
	
	csv = pd.read_csv ("dataset.csv",sep=",")
	csv = csv.sample (frac=1)
	csv = csv.head(10)
	md_column = []
	starting_time = time.time()
	pdfs = []
	
	for index, row in csv.iterrows():
		md_column.append (process_row(row))
	
		
		
	
	csv["markdown"] = md_column
	csv.to_csv("dataset_with_md.csv")
	print(time.time() - starting_time)

