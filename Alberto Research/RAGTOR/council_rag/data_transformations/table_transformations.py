from img2table.document import PDF
import base64
from groq import Groq
from pdf2image import convert_from_path

    
# extract tables from the list of pages of a pdf file
def extract_tables_from_page(pdf_path, list_pages):
    
    pdf = PDF(pdf_path,
                pages= list_pages,
                detect_rotation=False,
                pdf_text_extraction=True)
    extracted_tables = pdf.extract_tables()
    
    return extracted_tables

# convert pdf to images # expects the poppler path to be at the same level as the script
def convert_pdf_to_image(pdf_path, output_dir=r"/output_pdf_to_img/", poppler_path=r"poppler-24.08.0\Library\bin"):
    
    images = convert_from_path(pdf_path, poppler_path=poppler_path) 
    for i in range(len(images)):
        # Save pages as images in the pdf
        images[i].save(output_dir + pdf_path + '_page'+ str(i) +'.jpg', 'JPEG')
        
    return images

# encode image to base64 to be digestible by groq
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_multimodal_message_list(base_prompt, encoded_image):
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
    
    client = Groq(api_key=groq_key)
    encoded_image = encode_image(image_path)
    messages = format_multimodal_message_list(base_prompt, encoded_image)
    chat_completion = client.chat.completions.create(   messages=messages,
                                                        model=model
                                                        )
    
    return chat_completion.choices[0].message.content