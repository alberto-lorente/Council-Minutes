# Council-Minutes

- **Geothermal Project Classification:**
Our objective is to provide (automatically), for each document, its status in relation to a geothermal project. At the first level, a binary filter on the documents is requested:
  - concerns a geothermal project
  - unrelated to a geothermal project

We implemented a pydantic model that fulfills this returning a boolean response defined by the `GeothermalProjectDetection` class. The model classifies documents as either geothermal or non-geothermal

  ```python
# Binary Filter Pydantic Model

class GeothermalProjectDetection(BaseModel):
    is_geothermal: bool = Field(description="Whether the document concerns a geothermal project")
    chain_of_thought: str = Field(
        description="""Logical Steps that were taken to derive
        the final concluding statement"""
    )
  ```

- **Project Stage Classification:**
For documents that pass the binary filter, we classify them into different stages of project development using the `StageClassifier` model. This model categorizes documents into:
- idea/wish
- preliminary studies
- budget voted for definitive project
- implementation in progress
- installation completed

```python
class StageClassifier(BaseModel):
    stage: str = Field(description="The stage of the geothermal project")
    chain_of_thought: str = Field(
        description="""Logical Steps that were taken to derive
        the final concluding statement"""
    )
```

- **Data Extraction:**
For projects identified as geothermal, we extract key project metrics using specialized Pydantic models:

1. Initial Budget (`InitialBudget` model)
2. Final Cost (`FinalCost` model)
3. Estimated Duration (`EstimatedDuration` model)
4. Actual Duration (`ActualDuration` model)

Each model follows a similar pattern of providing both the extracted value and reasoning:

```python
class InitialBudget(BaseModel):
    answer: float = Field(description="Initial budget of the project")
    chain_of_thought: str = Field(
        description="""Logical Steps that were taken to derive
        the final concluding statement"""
    )
```

- **How Chain of Thought Reasoning is used:**
Each classification and extraction task includes chain-of-thought reasoning to explain how conclusions were reached. This is implemented through:

1. System messages that prompt the model to "generate a series of logical steps and reason about the problem"
2. Dedicated chain_of_thought fields in each Pydantic model
3. Temperature settings of 0.1 to ensure consistent reasoning
4. Clear prompts that guide the model through specific aspects to consider

Example system message:

```python
geothermal_detection_message = {
    "role": "system",
    "content": """
        Your task is to determine if the text concerns a geothermal project
        Consider mentions of:
        - Geothermal energy installations
        - Geothermal studies or surveys
        - Heat pump systems
        // ...existing code...
    """
}
```

- **Why Translation is used:**
Our Dataset consists of documents in French, and we need to translate them to English to perform the classification and extraction tasks. We used the `GoogleTranslator` to split long French text into chunks, translate them, and save both original and translated texts for verification.

- **Pipeline Aggregation:**
The notebook implements a complete pipeline that:

1. Loads documents from the dataset.csv file
2. Processes each document through multiple stages:
   - Translation (French to English)
   - Geothermal project detection
   - Project stage classification  
   - Data extraction (budget, duration, etc.)
3. Aggregates results into a structured JSON output

This is implemented in the `extract_project_details` function which coordinates all the processing steps and returns a comprehensive dictionary of results.

- **Final Output:**
The pipeline generates a JSON file containing:

1. Paths to original and translated texts
2. Binary geothermal classification result
3. Project stage classification
4. Extracted numerical data (budgets and durations)
5. Chain-of-thought reasoning for each decision

Example output structure:

```python
{
    "original_text_path": "path/to/french.txt",
    "english_translation_path": "path/to/english.txt",
    "is_geothermal": true,
    "geothermal_chain_of_thought": "reasoning...",
    "initial_budget": 100000.0,
    // ...additional fields...
}
```
