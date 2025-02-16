import torch
from transformers import RobertaTokenizerFast, EncoderDecoderModel, T5Tokenizer, T5ForConditionalGeneration
from groq import Groq

def generate_hf_summary(text, tokenizer, model, device):
    inputs = tokenizer([text], padding="max_length", return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_camembert_summary(text, device):
    
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/cam-roberta-large")
    model = EncoderDecoderModel.from_pretrained("microsoft/cam-roberta-large")
    summary = generate_hf_summary(text, tokenizer, model, device)
    
    return summary

def generate_t5_summary(text, device):
    tokenizer = T5Tokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
    model = T5ForConditionalGeneration.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
    summary = generate_hf_summary(text, tokenizer, model, device)
    return summary


def generate_groq_summary(base_summary_prompt, 
                        text_to_summarize, 
                        groq_key, 
                        model="llama-3.2-70b-versatile"):
        
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
