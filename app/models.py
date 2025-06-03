from pydantic import BaseModel
from typing import List

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