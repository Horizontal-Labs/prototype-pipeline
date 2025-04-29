from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import spacy
import torch
from typing import List, Dict, Any

# Initialize FastAPI app
app = FastAPI(title="Argument Mining API")

# Load NLP components
nlp = spacy.load("en_core_web_lg")

# Load the fine-tuned BERT model
tokenizer = BertTokenizer.from_pretrained("./argument-mining-model")
model = BertForSequenceClassification.from_pretrained("./argument-mining-model")

# Create argument component classifier pipeline
arg_classifier = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Define input/output models
class ArgumentMiningRequest(BaseModel):
    text: str

class ArgumentComponent(BaseModel):
    text: str
    type: str  # "claim", "premise", or "non-argument"
    confidence: float
    
class ArgumentRelation(BaseModel):
    source_idx: int
    target_idx: int
    relation_type: str  # "support", "attack", etc.
    confidence: float

class ArgumentMiningResponse(BaseModel):
    components: List[ArgumentComponent]
    relations: List[ArgumentRelation]
    
# Function to identify argument components
def identify_components(text: str) -> List[ArgumentComponent]:
    # Use spaCy to split text into sentences
    doc = nlp(text)
    components = []
    
    for sent in doc.sents:
        # Classify the sentence using our fine-tuned model
        result = arg_classifier(sent.text)[0]
        
        components.append(ArgumentComponent(
            text=sent.text,
            type=result["label"],
            confidence=result["score"]
        ))
    
    return components

# Function to identify relations between components
def identify_relations(components: List[ArgumentComponent]) -> List[ArgumentRelation]:
    # In a real system, this would use another model to predict relations
    # This is a simplified placeholder
    relations = []
    
    # Simple heuristic: look for keywords indicating relations
    for i, comp in enumerate(components):
        if comp.type == "claim":
            # Look for supporting premises
            for j, other in enumerate(components):
                if other.type == "premise":
                    if "because" in other.text.lower() or "since" in other.text.lower():
                        relations.append(ArgumentRelation(
                            source_idx=j,
                            target_idx=i,
                            relation_type="support",
                            confidence=0.8
                        ))
    
    return relations

# Define API endpoint
@app.post("/analyze", response_model=ArgumentMiningResponse)
async def analyze_arguments(request: ArgumentMiningRequest):
    try:
        # Process text to find argument components
        components = identify_components(request.text)
        
        # Identify relations between components
        relations = identify_relations(components)
        
        return ArgumentMiningResponse(
            components=components,
            relations=relations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}