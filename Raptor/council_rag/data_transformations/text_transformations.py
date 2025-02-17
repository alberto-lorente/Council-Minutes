import torch
from transformers import RobertaTokenizerFast, EncoderDecoderModel, T5Tokenizer, T5ForConditionalGeneration
from groq import Groq
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




def generate_hf_summary(text, tokenizer, model, device):
    """
    Not used anymore.
    """
    inputs = tokenizer([text], padding="max_length", return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_camembert_summary(text, device):
    """
    Not used anymore.
    """
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/cam-roberta-large")
    model = EncoderDecoderModel.from_pretrained("microsoft/cam-roberta-large")
    summary = generate_hf_summary(text, tokenizer, model, device)
    
    return summary

def generate_t5_summary(text, device):
    """
    Not used anymore.
    """
    tokenizer = T5Tokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
    model = T5ForConditionalGeneration.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
    summary = generate_hf_summary(text, tokenizer, model, device)
    return summary


def generate_groq_summary(base_summary_prompt, 
                        text_to_summarize, 
                        groq_key, 
                        model="gemma2-9b-it"):
    """
    Generates a summary of the text using Gemma2-9b-it with a base summary prompt with a {} to format with the text to summarize.
    Returns the summary response.
    """
    client = Groq(api_key=groq_key)
    summary_prompt = base_summary_prompt.format(text_to_summarize)
    messages = [
                {"role": "system",
                "content":  "Vous êtes un assistant utile" },
                {"role": "user",
                "content":  summary_prompt}
                ]
                        
    chat_completion = client.chat.completions.create(messages=messages, model=model)
        
    return chat_completion.choices[0].message.content
