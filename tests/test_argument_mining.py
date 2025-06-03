import pytest
from app.services import NLPService
from app.models import ArgumentComponent

@pytest.fixture
def nlp_service():
    service = NLPService(initialize=True)  # Initialize models for testing
    return service

def test_identify_argument_component(nlp_service, sample_text):
    components = nlp_service.identify_argument_components(sample_text)
    
    assert len(components) == 2
    assert all(isinstance(comp, ArgumentComponent) for comp in components)
    assert all(hasattr(comp, "text") for comp in components)
    assert all(hasattr(comp, "type") for comp in components)
    assert all(hasattr(comp, "confidence") for comp in components)
    assert all(0 <= comp.confidence <= 1 for comp in components)
    assert all(comp.type in ["claim", "premise", "non-argument"] for comp in components)

def test_identify_relations(nlp_service, sample_components):
    relations = nlp_service.identify_relations(sample_components)
    
    assert isinstance(relations, list)
    assert len(relations) > 0
    for relation in relations:
        assert hasattr(relation, "source_idx")
        assert hasattr(relation, "target_idx")
        assert hasattr(relation, "relation_type")
        assert hasattr(relation, "confidence")

def test_identify_argument_component_empty_text(nlp_service):
    components = nlp_service.identify_argument_components("")
    assert len(components) == 0

def test_identify_argument_component_single_sentence(nlp_service):
    test_text = "This is a test sentence."
    components = nlp_service.identify_argument_components(test_text)
    assert len(components) == 1
    assert isinstance(components[0], ArgumentComponent)

def test_identify_relations_no_components(nlp_service):
    relations = nlp_service.identify_relations([])
    assert len(relations) == 0

def test_identify_relations_single_component(nlp_service):
    components = [
        ArgumentComponent(text="This is a claim.", type="claim", confidence=0.9)
    ]
    relations = nlp_service.identify_relations(components)
    assert isinstance(relations, list) 