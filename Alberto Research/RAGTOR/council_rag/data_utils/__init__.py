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
    language="fr", 
    disable_ocr = True
)

def pdf2md(file_path):
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

def pf2mdpagebypage(input_pdf, output_folder, extract_text_function=pdf2md):
	doc = fitz.open(input_pdf)
	pages = []
	for i, page in enumerate(doc):
		starting_time = time.time()
		print(f"Processing page {i+1} of {len(doc)}")
		new_pdf = fitz.open()
		# new_pdf.insert_pdf(doc, from_page=i, to_page=i)
		
		output_filename = f"{output_folder}/page_{i+1}.pdf"
		# print(time.time() - starting_time)
		new_pdf.save(output_filename) 
		new_pdf.close()
		# print(time.time() - starting_time)
		pages.append(extract_text_function(output_filename)) # this is the part of the algo that takes very long with llama-parse  
	return pages

# took it out because it was not working
# def process_row (row, pagebypage=False):
# 	""" Process a line of the dataset """
# 	doc_id = row["doc_id"]
# 	cache_url = row["cache"]
# 	uuid_ = uuid.uuid1 ()
# 	pdf_path = "pdfs/" + str(uuid_)+".pdf"
# 	if download_pdf(cache_url, pdf_path):
#         if pagebypage:
#     		return pdf2mdpagebypage(pdf_path)
#         else:
#             return pdf2md (pdf_path)
        
# 	else:
# 		print("Could not process", doc_id)
# 		return None





