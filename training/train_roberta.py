import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from metrics import compute_metrics
from data_preperation import load_and_preprocess_data

# Create PyTorch datasets
class ArgumentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings, train_labels, eval_encodings, eval_labels = load_and_preprocess_data()

# Create datasets
train_dataset = ArgumentDataset(train_encodings, train_labels)
eval_dataset = ArgumentDataset(eval_encodings, eval_labels)

# Initialize model (I'd recommend DeBERTa-v3 or RoBERTa)
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",  # or "roberta-base"
    num_labels=3  # claim, premise, non-argument
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,  # You'll need to define this function
)



# Train the model
trainer.train()