from load import load_env_vars
from pydantic import BaseModel, computed_field
from encoding_utils import compute_embeddings

env_vars = load_env_vars()


class Doc(BaseModel):
    path: str
    
    def pdf_to_md(self) -> str:
        
        self.md = None
        
        return

    def md_to_txt(self)-> str:
        
        self.txt = None
        
        return

    def pdf_to_image(self)-> str: # IT WILL NOT BE A STR TYPE
        
        self.md = None
        
        return
    
################ COULD MOVE THIS TO DIFFERENT CLASSES ###############################################   
################ AND GET 1 CLASS FOR DATA TYPE        ###############################################
################ THAT WAY COMPUTING THE EMBEDDINGS AND THE SEARCH COULD BE DONE EASIER ##############

    def get_chunk_paras(self, 
                        n_sents_per_paragraph: int = 10) -> list:
        
        self.paras.n_sents = n_sents_per_paragraph
        self.paras = None
        
        return
    
    
    def get_individual_chunks(self, 
                            n_tokens_per_chunk: int = 350) -> list:
        
        self.chunks.n_tokens = n_tokens_per_chunk
        self.chunks = None
        
        return
    
    def compute_embeddings_for_doc(self)-> str:
        
        chunk_embeddings = compute_embeddings(model_type="EMBEDDINGS_OLLAMA_MODEL")
        tables_embeddings = compute_embeddings(model_type="VLM_OLLAMA_MODEL")
        para_summary_embeddings = compute_embeddings(self.paras, model_type="SUMMARY_OLLAMA_MODEL")
        
        return


if __name__ == "__main__":
    
    pass