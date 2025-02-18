from img2table.document import PDF
import base64
from groq import Groq
from pdf2image import convert_from_path
import os
import torch
from huggingface_hub import HfFolder, whoami

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


# extract tables from the list of pages of a pdf file
def extract_tables_from_page(pdf_path, list_pages=None):
    """
    Extract the tables from the pdf using the img2table library.
    Returns a dictionary with the extracted tables.
    """
    pdf = PDF(pdf_path,
                pages= list_pages,
                detect_rotation=False,
                pdf_text_extraction=True)
    extracted_tables = pdf.extract_tables()
    
    return extracted_tables

# convert pdf to images # expects the poppler path to be at the same level as the script
def convert_pdf_to_image(pdf_path, pdf_dir, output_dir="/output_pdf_to_img/", poppler_path=r"poppler-24.08.0\Library\bin"):
    """
    Convert the pdf to images using the pdf2image library.
    Saves the images in the output directory.
    Returns a list of the paths to the images.
    """
    output_dir_path = pdf_dir + output_dir
    pdf_name = pdf_path.split("\\")[-1]
    # print(output_dir_path)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    images_paths = []
    images = convert_from_path(pdf_path, poppler_path=poppler_path) 
    for i in range(len(images)):
        # Save pages as images in the pdf
        general_path = output_dir_path + pdf_name 
        # print(general_path)
        general_path = general_path.rstrip(".pdf")
        # print(general_path)
        image_path = general_path + '_page'+ str(i) +'.jpg'
        images[i].save(image_path, 'JPEG')
        
        images_paths.append(image_path)
    
    return images_paths


# encode image to base64 to be digestible by groq
def encode_image(image_path):
    """
    Encode the image to base64 to be digestible by groq.
    Returns the encoded image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_multimodal_message_list(base_prompt, encoded_image):
    """
    Formats the multimodal message list with the prompt and the encoded image.
    Returns the message list.
    """
    messages = [
        {"role": "user",
        "content": [
                {"type": "text", "text": base_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", # encoded image
                                                            },},
                    ],
        }
                ]
    return messages
    
def augment_multimodal_context(image_path, 
                                base_prompt, 
                                groq_key, 
                                model="llama-3.2-11b-vision-preview"):
    """
    Uses the encode_image function to encode the image and 
    then formats the multimodal message list with the format_multimodal_message_list function.
    Then it queries the VL Model.
    Returns the response.
    """
    client = Groq(api_key=groq_key)
    encoded_image = encode_image(image_path)
    messages = format_multimodal_message_list(base_prompt, encoded_image)
    chat_completion = client.chat.completions.create(   messages=messages,
                                                        model=model
                                                        )
    
    return chat_completion.choices[0].message.content

# format the table info into a dictionary
def format_table_info(list_pages_with_tables, list_table_htmls, table_descriptions):
    """
    Processes the table info into a dictionary from the dictionary of tables 
    and the VL Model response into a dictionary with the page number, the table description, the table html and the augmented context.
    Returns the list of processed tables dictionaries.
    """
    format_string = """{}
    
Tableau au format html:

{}"""

    list_processed_tables = []
    i = 0
    while i < len(list_table_htmls):
        
        html_table = list_table_htmls[i]
        page_num = list_pages_with_tables[i]
        descr_table = table_descriptions[i]
        table_string = format_string.format(descr_table, html_table)
        table_dict = {"page": page_num,
                    "table_context": descr_table,
                    "table_html": table_string,
                    "table_augmented_context": table_string}
        list_processed_tables.append(table_dict)
        i += 1
    return list_processed_tables