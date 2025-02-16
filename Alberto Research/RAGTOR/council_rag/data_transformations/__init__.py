import torch
from huggingface_hub import HfFolder, whoami

with open("HF_TOKEN.txt", "r") as f:
    hf_token = f.read()

with open("GROQ_KEY.txt", "r") as f:
    groq_token = f.read()

HfFolder.save_token(hf_token)
print(whoami()["name"])

device = "cpu"
if torch.cuda.is_available():
    print("Cuda available")
    device = torch.device('cuda')