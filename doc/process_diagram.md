# Argument Mining Pipeline Process Diagram
The process diagram above illustrates the sequential flow of data through the argument mining pipeline, showing how text is processed from initial input to final structured output.

:::Mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Preprocessor
    participant SpaCy
    participant BERT as BERT Model
    participant RelationClassifier
    participant Structurer

    Client->>FastAPI: POST /analyze with text
    
    FastAPI->>Preprocessor: Clean and normalize text
    Preprocessor-->>FastAPI: Preprocessed text
    
    FastAPI->>SpaCy: Process text with NLP pipeline
    SpaCy-->>FastAPI: Document with sentences
    
    loop For each sentence
        FastAPI->>BERT: Classify argument component
        BERT-->>FastAPI: Component type & confidence
    end
    
    FastAPI->>RelationClassifier: Identify relations between components
    RelationClassifier-->>FastAPI: Support/Attack relations
    
    FastAPI->>Structurer: Organize into coherent argument structure
    Structurer-->>FastAPI: Structured argument
    
    FastAPI-->>Client: JSON response with components and relations
    
    note over Client,FastAPI: Optional validation
    Client->>FastAPI: POST /validate with expected components
    FastAPI-->>Client: Validation metrics
:::

## Process Flow Explanation

1. Client Request

The process begins when a client sends a POST request to the /analyze endpoint with text content.


2. Text Preprocessing

The FastAPI server forwards the text to the Preprocessor.
The Preprocessor cleans the text (removing excess whitespace, normalizing characters, etc.) and returns it.


3. NLP Processing

The preprocessed text is sent to SpaCy for linguistic analysis.
SpaCy returns a document with identified sentences, tokens, and linguistic features.


4. Argument Component Detection

For each sentence, the BERT model analyzes and classifies it as a claim, premise, or non-argument.
Each classification includes a confidence score.


5. Relation Classification

The RelationClassifier examines pairs of components to identify support or attack relationships.
This determines which premises support which claims and any counter-arguments.


6. Argument Structuring

The Structurer organizes the components and relations into a coherent argument structure.
This could include grouping related components and establishing hierarchy.


7. Response

The FastAPI server returns a JSON response with the structured argument data.


8. Optional Validation

For testing or quality assurance, the client can send expected component data.
The server returns validation metrics comparing predictions with expected values.



This process flow demonstrates how the system transforms raw text into structured argumentation data through a series of specialized processing steps.