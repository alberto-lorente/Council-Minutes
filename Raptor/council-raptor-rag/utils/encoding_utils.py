import os
from load import load_env_vars 
import ollama 
import torch

env_vars = load_env_vars()

def compute_embeddings( text: str, 
                        model: str, # I DONT THINK I WILL NEED THIS ONE
                        model_type: str = "EMBEDDINGS_OLLAMA_MODEL") -> torch.Tensor:
    
    assert model_type in env_vars.keys()
    model_id = env_vars[model_type]
    
    return

if __name__ == "__main__":
    
    text  = "THIS IS SOME EXAMPLE TEXT"
    emb = compute_embeddings(text)
    
    print(type(emb))
    print(emb.shape)
    print(emb)