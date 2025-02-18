## Repository Overview

This part of the repository contains the code for the Raptor side of the project.
The folder `council_rag` contains the modules we developed to perform our experiments and is documented.

The folder `Evaluation data` contains the pdfs and the script we used to evaluate the performance of the summary and table augmentation model as well as the evaluation data.

`poppler` is required a required dependency to convert the pdfs to markdown. Do not delete it!

The jupyter notebooks show the final steps we followed to develop the pipeline cleaned up and you should be able to run them.

## Raptor Pipeline Flow

1. **Pre-processing**
   - Reads markdown and PDF documents
   - Splits text into sentences using spaCy (French language model)
   - Groups sentences into paragraphs
   - Generates embeddings using Gemma 2B model
   - Performs clustering using Gaussian Mixture Models to identify related content

2. **Data Transformations**
To deal with tables:
   - Extracts tables from PDF pages
   - Converts PDF pages to images
   - Encodes images for multimodal context
   - Generates table information with a VL Model
To deal with longer context:
   - Generates summaries for the clusters

3. **RAG Pipeline**
Diagram of the RAG query flow:
![OVERVIEW QUERY](https://github.com/user-attachments/assets/a559f7e0-62db-455f-859f-86b27a53eb10)

- At a first step, we query the cluster summaries.
- Then we query those chunks which belonged to the cluster returned in the previous step as well as the tables.
- This information is formated together for the augmented generation.


### Requirements

In order to run the code, you need to have the following files:
- `HF_TOKEN.txt` - Hugging Face API token
- `GROQ_KEY.txt` - GROQ API key
- A Llama Parse API key if your pdfs have not been converted to markdown yet.

Install the requirements:
```bash
pip install -r req![OVERVIEW QUERY](https://github.com/user-attachments/assets/e9c970b6-5b14-496b-991d-c201dcfd8a36)
uirements.txt
```
And check that you comply with the other requirements file. Note: your kernel will crash if CUDA is not available.
