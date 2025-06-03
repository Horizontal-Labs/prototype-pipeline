from fastapi import APIRouter, HTTPException
from .models import ArgumentMiningRequest, ArgumentMiningResponse
from .services import NLPService

router = APIRouter()
nlp_service = NLPService.get_instance()

@router.post("/analyze", response_model=ArgumentMiningResponse)
async def analyze_arguments(request: ArgumentMiningRequest):
    """
    Analyze text for argument components and relations.
    
    Args:
        request: ArgumentMiningRequest containing the text to analyze
        
    Returns:
        ArgumentMiningResponse containing identified components and relations
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Process text to find argument components
        components = nlp_service.identify_argument_components(request.text)
        
        # Identify relations between components
        relations = nlp_service.identify_relations(components)
        
        return ArgumentMiningResponse(
            components=components,
            relations=relations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy"} 