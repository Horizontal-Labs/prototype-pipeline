import os
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from typing import List, Tuple, Optional
from .models import ArgumentComponent, ArgumentRelation

class NLPService:
    _instance: Optional['NLPService'] = None
    
    def __init__(self, initialize: bool = True):
        if initialize:
            self.initialize()

    def initialize(self):
        """Initialize models and services."""
        self.nlp = self._load_spacy_model()
        self.tokenizer, self.model = self._load_bert_model()
        self.arg_classifier = self._create_pipeline()

    @classmethod
    def get_instance(cls) -> 'NLPService':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_spacy_model(self) -> spacy.language.Language:
        """Load the spaCy model with fallback to smaller model."""
        try:
            return spacy.load("en_core_web_lg")
        except OSError:
            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "No spaCy model found. Please install one using:\n"
                    "python -m spacy download en_core_web_lg\n"
                    "or\n"
                    "python -m spacy download en_core_web_sm"
                )

    def _load_bert_model(self) -> Tuple[BertTokenizer, BertForSequenceClassification]:
        """Load the BERT model and tokenizer."""
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not found in environment variables. "
                "Please add your Hugging Face token to the .env file."
            )
        
        try:
            tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                use_auth_token=hf_token
            )
            model = BertForSequenceClassification.from_pretrained(
                "google-bert/bert-base-uncased",
                num_labels=3,  # claim, premise, non-argument
                use_auth_token=hf_token
            )
            return tokenizer, model
        except Exception as e:
            raise RuntimeError(f"Error loading BERT model: {str(e)}")

    def _create_pipeline(self):
        """Create the argument classification pipeline."""
        return pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def _map_label_to_type(self, label: str) -> str:
        """Map BERT output labels to argument types."""
        label_map = {
            "LABEL_0": "claim",
            "LABEL_1": "premise",
            "LABEL_2": "non-argument"
        }
        return label_map.get(label, "unknown")

    def identify_argument_components(self, text: str) -> List[ArgumentComponent]:
        """Identify argument components in the given text."""
        doc = self.nlp(text)
        components = []
        
        for sent in doc.sents:
            result = self.arg_classifier(sent.text)[0]
            components.append(ArgumentComponent(
                text=sent.text,
                type=self._map_label_to_type(result["label"]),
                confidence=result["score"]
            ))
        
        return components

    def identify_relations(self, components: List[ArgumentComponent]) -> List[ArgumentRelation]:
        """Identify relations between argument components."""
        relations = []
        
        for i, comp in enumerate(components):
            if comp.type == "claim":
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

# Create a global instance of the NLP service
nlp_service = NLPService.get_instance() 