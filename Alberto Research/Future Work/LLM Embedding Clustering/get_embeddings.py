# Workflow for all the Document Classes

# Replicating the workflow of the example for the rest of the data.
# - Compute Word Embeddings.

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

data_path = "../../dataset_full_texts.csv"
data = pd.read_csv(data_path)

print("Data Loaded")

# Getting Embeddings

model =  SentenceTransformer("dangvantuan/sentence-camembert-base")
print("Model Loaded")

data["embeddings"] = data["extracted_text"].apply(lambda text: model.encode(text))
print("Text Embeddings Computed")

data.to_csv(data_path, index=False)
print("Embeddings saved")