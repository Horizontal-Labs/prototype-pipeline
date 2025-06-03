import pytest
from transformers import BertTokenizer, BertForSequenceClassification
import spacy
import torch
from app.services import NLPService

@pytest.fixture
def nlp_service():
    service = NLPService(initialize=True)
    return service

def test_spacy_model_loaded(nlp_service):
    assert nlp_service.nlp is not None
    assert isinstance(nlp_service.nlp, spacy.language.Language)
    
def test_bert_tokenizer_loaded(nlp_service):
    assert nlp_service.tokenizer is not None
    assert isinstance(nlp_service.tokenizer, BertTokenizer)

def test_bert_model_loaded(nlp_service):
    assert nlp_service.model is not None
    assert isinstance(nlp_service.model, BertForSequenceClassification)

def test_pipeline_loaded(nlp_service):
    assert nlp_service.arg_classifier is not None
    assert callable(nlp_service.arg_classifier)

def test_model_device(nlp_service):
    expected_device = 0 if torch.cuda.is_available() else -1
    # Check if model is on the correct device
    if expected_device == 0:
        assert next(nlp_service.model.parameters()).is_cuda
    else:
        assert not next(nlp_service.model.parameters()).is_cuda

def test_tokenizer_vocabulary(nlp_service):
    # Test if tokenizer has a reasonable vocabulary size
    assert len(nlp_service.tokenizer.vocab) > 1000

def test_model_output_dimensions(nlp_service):
    # Test if model outputs correct number of classes
    test_input = nlp_service.tokenizer("This is a test input", return_tensors="pt")
    with torch.no_grad():
        outputs = nlp_service.model(**test_input)
    assert outputs.logits.shape[1] in [2, 3]  # Assuming 2 or 3 classes 