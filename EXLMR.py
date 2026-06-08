import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
import os
import pandas as pd

# Paths
new_vocab_path = "/home/teklehaymanot/EXLMR/vocab.json"  # New vocabulary file
train_path = "/home/teklehaymanot/EXLMR/train.csv"  # Training data
test_path = "/home/teklehaymanot/EXLMR/test.csv"  # Test data
model_name = "xlm-roberta-base"  # Pre-trained XLM-R model
output_dir = "/home/teklehaymanot/EXLMR_Model"  # Directory to save the updated model
finetuned_model_dir = "/home/teklehaymanot/finetuned_model"  # Directory to save the fine-tuned model

# Disable WandB
os.environ['WANDB_DISABLED'] = 'true'

# Load the new vocabulary from vocab.json
with open(new_vocab_path, "r", encoding="utf-8") as vocab_file:
    new_vocab = json.load(vocab_file)

# Extract new tokens from the vocab.json
new_tokens = list(new_vocab.keys())

# Load the pre-trained tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Add new tokens to the tokenizer
num_added_tokens = tokenizer.add_tokens(new_tokens)

# Resize the model's embeddings to accommodate the new tokens
model.resize_token_embeddings(len(tokenizer))

# Define new_token_ids
new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)

# Initialize new embeddings using the mixed strategy
with torch.no_grad():
    for token_id in new_token_ids[-num_added_tokens:]:  # Only update new token embeddings
        # Method 1: Random Initialization
        random_init = torch.nn.init.normal_(torch.empty(model.config.hidden_size))
        
        # Method 2: Mean of existing embeddings
        existing_embeddings = model.roberta.embeddings.word_embeddings.weight[:-num_added_tokens, :]
        mean_embedding = existing_embeddings.mean(dim=0)
        
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
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir=f"{finetuned_model_dir}/logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=None  # Disable reporting to WandB
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(finetuned_model_dir)
tokenizer.save_pretrained(finetuned_model_dir)

print(f"Fine-tuned model saved to {finetuned_model_dir}")
