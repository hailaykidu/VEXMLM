import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Paths
new_vocab_path = "/homes/neumann/teklehaymanot/EXLMR/vocab.json"  # New vocabulary file
train_path = "/homes/neumann/teklehaymanot/EXLMR/cleaned_train.csv"  # Training data
test_path = "/homes/neumann/teklehaymanot/EXLMR/cleaned_test.csv"  # Test data
model_name = "xlm-roberta-base"  # Pre-trained XLM-R model
output_dir = "/homes/neumann/teklehaymanot/EXLMR_Model"  # Directory to save the updated model
finetuned_model_dir = "/homes/neumann/teklehaymanot/finetuned_model"  # Directory to save the fine-tuned model
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

# Disable WandB
os.environ['WANDB_DISABLED'] = 'true'

# Load the new vocabulary from vocab.json
with open(new_vocab_path, "r", encoding="utf-8") as vocab_file:
    new_vocab = json.load(vocab_file)

# Extract new tokens from the vocab.json
new_tokens = list(new_vocab.keys())

# Load the pre-trained tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

# Add new tokens to the tokenizer. Capture the vocab size *before* adding, so we can
# identify exactly which token IDs are newly added below -- add_tokens() silently skips
# tokens that already exist in the vocab, so its return value (a count) is not enough
# to know *which* entries in `new_tokens` were actually added.
original_vocab_size = len(tokenizer)
num_added_tokens = tokenizer.add_tokens(new_tokens)

# Resize the model's embeddings to accommodate the new tokens
model.resize_token_embeddings(len(tokenizer))

# Newly added tokens are always appended with sequential IDs starting at
# original_vocab_size, regardless of where they appeared in new_tokens or how many
# duplicates were skipped -- this is the only correct way to identify them.
new_token_ids = list(range(original_vocab_size, len(tokenizer)))
assert len(new_token_ids) == num_added_tokens

# Initialize new embeddings using the mixed strategy
with torch.no_grad():
    existing_embeddings = model.roberta.embeddings.word_embeddings.weight[:original_vocab_size, :]
    mean_embedding = existing_embeddings.mean(dim=0)

    for token_id in new_token_ids:  # Only update new token embeddings
        # Method 1: Random Initialization
        random_init = torch.nn.init.normal_(torch.empty(model.config.hidden_size))

        # Method 2: Mean of existing embeddings (computed once, outside the loop)

        # Mixed strategy: Average of random and mean embeddings
        mixed_embedding = (random_init + mean_embedding) / 2

        # Assign the mixed embedding to the new token
        model.roberta.embeddings.word_embeddings.weight[token_id] = mixed_embedding

# Save the updated tokenizer and model
os.makedirs(output_dir, exist_ok=True)
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

print(f"Updated model and tokenizer with {num_added_tokens} new tokens have been saved to {output_dir}")

# Define the dataset class
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract label and sentence based on the provided format
        label = int(self.data.iloc[index, 0])  # First column is the label
        sentence = str(self.data.iloc[index, 1])  # Second column is the text
        
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load datasets with semicolon delimiter
try:
    train_df = pd.read_csv(train_path, delimiter=';', header=None, on_bad_lines='skip')
    test_df = pd.read_csv(test_path, delimiter=';', header=None, on_bad_lines='skip')
    
    # Print the first few rows to check if the data is loaded correctly
    print("Train DataFrame head:", train_df.head())
    print("Test DataFrame head:", test_df.head())

except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")

# Set parameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-5

# Create datasets
train_dataset = SentimentDataset(train_df, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_df, tokenizer, MAX_LEN)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir=finetuned_model_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir=f"{finetuned_model_dir}/logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"  # Disable reporting to WandB
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# If this run was preempted/requeued, resume model weights from the last checkpoint.
# Note: optimizer/scheduler state (optimizer.pt) is intentionally NOT restored here --
# transformers refuses to torch.load() it on torch < 2.6 (CVE-2025-32434), which is
# what's installed in this env. Training restarts its LR schedule from this point on,
# but the learned weights themselves are not lost.
last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
if last_checkpoint is not None:
    state_dict = load_file(os.path.join(last_checkpoint, "model.safetensors"))
    model.load_state_dict(state_dict)
    print(f"Resumed model weights from {last_checkpoint}")

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(finetuned_model_dir)
tokenizer.save_pretrained(finetuned_model_dir)

print(f"Fine-tuned model saved to {finetuned_model_dir}")
