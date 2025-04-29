import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from pathlib import Path

def load_and_preprocess_data():
    """
    Load and preprocess the dataset for argument mining.
    
    Returns:
        train_encodings: Encoded training data.
        train_labels: Labels for training data.
        eval_encodings: Encoded evaluation data.
        eval_labels: Labels for evaluation data.
    """
    # Load the dataset^
    path = Path(__file__).parent.joinpath('datasets', 'persuade', 'persuade_corpus_1.0.csv')
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please check the path.")
    df = pd.read_csv('datasets/persuade/persuade_v2.csv')  # Adjust path as needed

    # Split data
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize tokenizer (RoBERTa often performs well on argument mining)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Prepare the data for component classification
    def preprocess_for_components(dataframe):
        texts = dataframe['text'].tolist()  # Adjust column name based on actual data
        # Map labels to IDs (claim=0, premise=1, non-argument=2)
        label_map = {"claim": 0, "premise": 1, "non-argument": 2}
        labels = [label_map[label] for label in dataframe['component_type']]
        
        # Tokenize texts
        encodings = tokenizer(texts, truncation=True, padding=True)
        
        return encodings, labels

    train_encodings, train_labels = preprocess_for_components(train_df)
    eval_encodings, eval_labels = preprocess_for_components(eval_df)
    return train_encodings, train_labels, eval_encodings, eval_labels