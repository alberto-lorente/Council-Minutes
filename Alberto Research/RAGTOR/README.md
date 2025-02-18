## Repository Overview

This part of the repository contains the code for the Raptor side of the project.
The folder `council_rag` contains the modules we developed to perform our experiments and is documented.

The folder `Evaluation data` contains the pdfs and the script we used to evaluate the performance of the summary and table augmentation model as well as the evaluation data.

`poppler` is required a required dependency to convert the pdfs to markdown. Do not delete it!

The jupyter notebooks show the final steps we followed to develop the pipeline cleaned up and you should be able to run them.

## Comments on the Development

Some comments on specific matters regarding the project:

   - We tried a variety of french models from the MTEB(fra, v1) BenchMark but at the end they were all too heavy. We settled for Gemma since it is the only one that we could run locally.
   - Throughout the modules/notebooks we are using a naive check length method to compute string lengths. We acknowledge that this is not ideal, but comparing the token segmentation of spacy with this naive method, we found that the final number of tokens was very similar.
   - Most Groq models have a low token per minute limite so we are using the one which has the highest one. It would be fairly difficult to surpass it but just in case, we have added a naive token track and sleep functionality when we call it just in case.
   - Ideally the min number of clusters would be computed dynamically (since as the text length grows, the clusters will be bigger and we may run into LLM querying limits) but here we are setting them manually as an argument to our main preprocessing function. As a rule of thumb, 8 clusters works fine for documents of around 40-50 pages. 

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
pip install -r requirements.txt
```
And check that you comply with the other requirements file. Note: your kernel will crash if CUDA is not available.
