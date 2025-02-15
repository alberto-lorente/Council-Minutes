-- Creating the vector extension to store the embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Creating the main document table
CREATE TABLE doc_info (
    doc_id              VARCHAR(80)     NOT NULL    PRIMARY KEY, --fetched from the pdf file
    n_pages             SMALLINT        NOT NULL,                -- number of total pages in the doc
    n_tokens            INT             NOT NULL,                -- number of tokens in the doc, calculated by the naive length function
    full_text           TEXT            NOT NULL,                -- full text of the document
    );

-- Creating the table for the chunks
CREATE TABLE chunks (
    doc_id              VARCHAR(80)     NOT NULL,     -- string of var len with up to 80 characters
    chunk_id            BIGSERIAL       NOT NULL,     -- string of var len with up to 80 characters
    chunk_text          TEXT            NOT NULL,     -- text of the chunk
    chunk_embedding     VECTOR(768)     NOT NULL,     -- embedding of the chunk
    chunk_type          VARCHAR(20)     NOT NULL,     -- is the chunk a sentence, a paragraph, a summary, a cluster
    chunk_cluster       SMALLINT        NOT NULL,     -- cluster of the chunk
    page_ns             VECTOR(768)     NOT NULL,     -- page number(s) where the chunk appears, could be more than one if it spans multiple pages
    has_table           BOOLEAN         NOT NULL    DEFAULT FALSE,       -- true if the chunk contains a table

    PRIMARY KEY (chunk_id, doc_id)
);

-- Creating the table for the tables in the documents
CREATE TABLE tables (
    doc_id                  VARCHAR(80)     NOT NULL,   
    chunk_id                BIGSERIAL       NOT NULL,   -- id of the chunk that contains the table
    table_id                BIGSERIAL       NOT NULL,   -- id of the table
    page_n                  SMALLINT        NOT NULL,   -- page number(s) where the table appears
    markdown_text           TEXT            ,           -- text of the table in markdown format
    markdown_text_embedding VECTOR(768)     ,           -- embedding of the markdown text
    html_text               TEXT            ,           -- text of the table in htm format
    generated_description   TEXT            ,           -- generated description of the table
    table_descr_embedding   VECTOR(768)     ,           -- embedding of the generated description
    generation_model_id     VARCHAR(20)     ,           -- id of the model used to generate the description
    
    PRIMARY KEY (doc_id, chunk_id, table_id)
);

-- Creating the table for the models
CREATE TABLE generation_models (
    generation_model_id   SERIAL        NOT NULL    PRIMARY KEY,   
    generation_model_name VARCHAR(20)   NOT NULL
);

