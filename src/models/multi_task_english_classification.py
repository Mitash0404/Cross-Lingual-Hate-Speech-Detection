# ===============================================
# Multi-Task Learning Pipeline: Hate, Offensive, Sentiment (2 Classes)
# ===============================================

# ---------------------------
# 1. Imports and Setup
# ---------------------------

import os
import re
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaModel, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# ---------------------------
# 2. Device Configuration
# ---------------------------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# ---------------------------
# 3. Define Directories
# ---------------------------

DATA_DIR = './data/english/'       # Directory containing English datasets
HINDI_DATA_DIR = './data/hindi/'   # Directory containing Hindi datasets
MODEL_DIR = './models/multi_task_english/'  # Directory to save models
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# 4. Helper Functions
# ---------------------------

def clean_text(text):
    """
    Cleans the input text by removing URLs, mentions, hashtags, and extra whitespaces.
    """
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def balance_dataset(df, target_labels, samples_per_class):
    """
    Balances the dataset by sampling a specified number of samples for each target label.
    """
    balanced_df = pd.DataFrame()
    for label in target_labels:
        label_df = df[df['label'] == label]
        available_samples = len(label_df)
        if available_samples < samples_per_class:
            raise ValueError(f"Not enough samples for label '{label}'. Required: {samples_per_class}, Available: {available_samples}")
        sampled_df = label_df.sample(n=samples_per_class, random_state=42)
        balanced_df = pd.concat([balanced_df, sampled_df], ignore_index=True)
    return balanced_df

# ---------------------------
# 5. Load and Preprocess Datasets
# ---------------------------

# ---------------------------
# 5.1. Load Hate Speech and Offensive Language Dataset
# ---------------------------

hate_dataset_path = os.path.join(DATA_DIR, 'hate_offn_data.csv')  # Update with your actual file path

if not os.path.exists(hate_dataset_path):
    raise FileNotFoundError(f"Hate Speech dataset not found at {hate_dataset_path}. Please download it from https://github.com/t-davidson/hate-speech-and-offensive-language and place it in the specified directory.")

# Load hate speech data
hate_df = pd.read_csv(hate_dataset_path)
# Expected columns: 'tweet', 'class'

# Rename columns for consistency
hate_df.rename(columns={'tweet': 'text', 'class': 'label'}, inplace=True)

# Clean text
hate_df['clean_text'] = hate_df['text'].apply(clean_text)

# Display sample
print("\nHate Speech Dataset Sample:")
print(hate_df.head())

# Display label distributions
print("\nHate Speech Label Distribution:")
print(hate_df['label'].value_counts())

# Define target labels for Hate and Offensive tasks
hate_target_labels = ['HATE', 'NOT']
offensive_target_labels = ['OFFN', 'NOT']

# Define number of samples per class
samples_per_class_hate = 2500
samples_per_class_offensive = 2500

# Balance Hate Speech Dataset for 'HATE' and 'NOT'
balanced_hate_df = balance_dataset(hate_df, hate_target_labels, samples_per_class_hate)

# Balance Offensive Dataset for 'OFFN' and 'NOT'
if 'OFFN' in hate_df['label'].unique():
    balanced_offensive_df = balance_dataset(hate_df[hate_df['label'].isin(['OFFN', 'NOT'])], offensive_target_labels, samples_per_class_offensive)
    print("\nBalanced Offensive Dataset Label Distribution:")
    print(balanced_offensive_df['label'].value_counts())
else:
    balanced_offensive_df = pd.DataFrame(columns=hate_df.columns)
    print("\nNo 'OFFN' labels found in Hate Speech dataset.")
    
# Balance Hate Speech Dataset
print("\nBalanced Hate Speech Dataset Label Distribution:")
print(balanced_hate_df['label'].value_counts())

# ---------------------------
# 5.2. Load Sentiment140 Dataset (2 Classes)
# ---------------------------

sentiment140_path = os.path.join(DATA_DIR, 'sentiment140.csv')  # Update with your actual file path

if not os.path.exists(sentiment140_path):
    raise FileNotFoundError(f"Sentiment140 dataset not found at {sentiment140_path}. Please download it from https://www.kaggle.com/kazanova/sentiment140 and place it in the specified directory.")

# Load Sentiment140 data
sentiment140_df = pd.read_csv(sentiment140_path, encoding='latin-1', header=None)
# Columns: 0-5, where column 0 is the sentiment

# Assign column names
sentiment140_df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Map sentiment labels (2 classes: Negative and Positive)
sentiment_map = {0: 'Negative', 4: 'Positive'}
sentiment140_df['mapped_label'] = sentiment140_df['sentiment'].map(sentiment_map)

# Create 'label' column to match the balance_dataset function
sentiment140_df['label'] = sentiment140_df['mapped_label']

# Clean text
sentiment140_df['clean_text'] = sentiment140_df['text'].apply(clean_text)

# Display sample
print("\nSentiment140 Dataset Sample:")
print(sentiment140_df.head())

# Display label distributions
print("\nSentiment140 Label Distribution:")
print(sentiment140_df['mapped_label'].value_counts())

# Define target labels for Sentiment task
sentiment_target_labels = ['Negative', 'Positive']

# Define number of samples per class
samples_per_class_sentiment = 2500  # 2500 Negative, 2500 Positive

# Balance Sentiment Dataset
balanced_sentiment140_df = balance_dataset(sentiment140_df, sentiment_target_labels, samples_per_class_sentiment)

print("\nBalanced Sentiment140 Dataset Label Distribution:")
print(balanced_sentiment140_df['mapped_label'].value_counts())

# ---------------------------
# 6. Define Dataset Classes
# ---------------------------

class MultiTaskDataset(Dataset):
    def __init__(self, hate_df, offensive_df, sentiment_df, tokenizer, max_len):
        self.data = []
        
        # Process hate speech data
        for _, row in hate_df.iterrows():
            self.data.append({
                'text': row['clean_text'],
                'task': 'hate',
                'label': 1 if row['label'] == 'HATE' else 0
            })
        
        # Process offensive language data
        if not offensive_df.empty:
            for _, row in offensive_df.iterrows():
                self.data.append({
                    'text': row['clean_text'],
                    'task': 'offensive',
                    'label': 1 if row['label'] == 'OFFN' else 0
                })
        
        # Process sentiment data
        sentiment_map = {'Negative': 0, 'Positive': 1}
        for _, row in sentiment_df.iterrows():
            self.data.append({
                'text': row['clean_text'],
                'task': 'sentiment',
                'label': sentiment_map.get(row['mapped_label'], 1)  # Default to 'Positive' if not found
            })
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'task': item['task'],
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# ---------------------------
# 7. Initialize Tokenizer and Dataset
# ---------------------------

# Initialize Tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
MAX_LEN = 128

# Instantiate the multi-task dataset
multitask_dataset = MultiTaskDataset(
    hate_df=balanced_hate_df,
    offensive_df=balanced_offensive_df,
    sentiment_df=balanced_sentiment140_df,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# ---------------------------
# 8. Split Dataset into Train and Test
# ---------------------------

train_size = 0.9
train_data, test_data = train_test_split(
    multitask_dataset.data,
    test_size=1 - train_size,
    random_state=42,
    stratify=[item['task'] for item in multitask_dataset.data]
)

# Define Split Dataset Class
class SplitMultiTaskDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'task': item['task'],
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# Create Training and Testing Datasets
train_dataset = SplitMultiTaskDataset(train_data, tokenizer, MAX_LEN)
test_dataset = SplitMultiTaskDataset(test_data, tokenizer, MAX_LEN)

# ---------------------------
# 9. Create DataLoaders with Optimizations
# ---------------------------

BATCH_SIZE = 32  # Adjust based on GPU memory
NUM_WORKERS = 32  # Limit workers to 32 to avoid warnings

# Define Collate Function for Dynamic Padding
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    tasks = [item['task'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'task': tasks,
        'labels': torch.stack(labels)
    }

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn  # Use dynamic padding
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn  # Use dynamic padding
)

# ---------------------------
# 10. Define the Multi-Task Model
# ---------------------------

class MultiTaskXLMR(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MultiTaskXLMR, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Define separate classification heads
        self.classifiers = nn.ModuleDict({
            'hate': nn.Linear(hidden_size, num_labels['hate']),
            'offensive': nn.Linear(hidden_size, num_labels['offensive']),
            'sentiment': nn.Linear(hidden_size, num_labels['sentiment'])
        })
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, task):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifiers[task](cls_output)
        return logits

# Define number of labels for each task
num_labels = {
    'hate': 2,        # 'NOT', 'HATE'
    'offensive': 2,   # 'NOT', 'OFFN'
    'sentiment': 2    # 'Negative', 'Positive'
}

# Initialize the model
model = MultiTaskXLMR('xlm-roberta-base', num_labels)

# Parallelize the model if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)

model = model.to(device)

# ---------------------------
# 11. Define Training and Evaluation Functions
# ---------------------------

def compute_weights(labels, num_classes):
    """
    Computes class weights for handling class imbalance.
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float).to(device)

# Extract labels for each task from training data
labels_hate = [item['label'] for item in train_data if item['task'] == 'hate']
labels_offensive = [item['label'] for item in train_data if item['task'] == 'offensive']
labels_sentiment = [item['label'] for item in train_data if item['task'] == 'sentiment']

# Compute class weights
weight_hate = compute_weights(labels_hate, num_labels['hate'])
weight_offensive = compute_weights(labels_offensive, num_labels['offensive'])
weight_sentiment = compute_weights(labels_sentiment, num_labels['sentiment'])

# Define loss functions with class weights
loss_functions = {
    'hate': nn.CrossEntropyLoss(weight=weight_hate),
    'offensive': nn.CrossEntropyLoss(weight=weight_offensive),
    'sentiment': nn.CrossEntropyLoss(weight=weight_sentiment)
}

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
EPOCHS = 5
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warm-up
    num_training_steps=total_steps
)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

def train_epoch(model, loader, optimizer, scheduler, loss_funcs, scaler):
    """
    Trains the model for one epoch using mixed precision.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tasks = batch['task']
        labels = batch['labels'].to(device)
        
        # Group samples by task
        tasks_unique = set(tasks)
        batch_loss = 0
        with autocast():
            for task in tasks_unique:
                idx = [i for i, t in enumerate(tasks) if t == task]
                if not idx:
                    continue
                task_input_ids = input_ids[idx]
                task_attention_mask = attention_mask[idx]
                task_labels = labels[idx]
                
                # Forward pass
                logits = model(input_ids=task_input_ids, attention_mask=task_attention_mask, task=task)
                loss = loss_funcs[task](logits, task_labels)
                batch_loss += loss
        
        # Backward pass with mixed precision
        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += batch_loss.item()
    return total_loss / len(loader)

def eval_model(model, loader, loss_funcs):
    """
    Evaluates the model on the validation/test dataset.
    """
    model.eval()
    total_loss = 0
    all_labels = {'hate': [], 'offensive': [], 'sentiment': []}
    all_preds = {'hate': [], 'offensive': [], 'sentiment': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tasks = batch['task']
            labels = batch['labels'].to(device)
            
            tasks_unique = set(tasks)
            batch_loss = 0
            for task in tasks_unique:
                idx = [i for i, t in enumerate(tasks) if t == task]
                if not idx:
                    continue
                task_input_ids = input_ids[idx]
                task_attention_mask = attention_mask[idx]
                task_labels = labels[idx]
                
                # Forward pass
                logits = model(input_ids=task_input_ids, attention_mask=task_attention_mask, task=task)
                loss = loss_funcs[task](logits, task_labels)
                batch_loss += loss.item()
                
                # Predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds[task].extend(preds)
                all_labels[task].extend(task_labels.cpu().numpy())
            
            total_loss += batch_loss
    
    avg_loss = total_loss / len(loader)
    
    # Calculate metrics
    metrics = {}
    for task in all_labels:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels[task], all_preds[task], average='weighted', zero_division=0
        )
        metrics[task] = {'precision': precision, 'recall': recall, 'f1': f1}
    return avg_loss, metrics

# ---------------------------
# 12. Training Loop
# ---------------------------

# Initialize best F1 scores
best_f1 = {'hate': 0, 'offensive': 0, 'sentiment': 0}

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 30)
    
    # Training
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_functions, scaler)
    print(f'Train Loss: {train_loss:.4f}')
    
    # Evaluation
    val_loss, val_metrics = eval_model(model, test_loader, loss_functions)
    print(f'Validation Loss: {val_loss:.4f}')
    for task, metrics in val_metrics.items():
        print(f"{task.capitalize()} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Save best model per task
        if metrics['f1'] > best_f1[task]:
            best_f1[task] = metrics['f1']
            # Handle DataParallel by saving model.module.state_dict() if necessary
            save_path = os.path.join(MODEL_DIR, f'best_model_{task}.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Best model for task '{task}' saved at {save_path}.")
    print()

# ---------------------------
# 13. Save the Final Model and States
# ---------------------------

# Save the final model
final_model_path = os.path.join(MODEL_DIR, 'multi_task_xlm_roberta.pth')
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), final_model_path)
else:
    torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

# Save the tokenizer
tokenizer_save_path = os.path.join(MODEL_DIR, 'tokenizer_xlm_roberta')
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Tokenizer saved to {tokenizer_save_path}")

# Save the optimizer and scheduler states
optimizer_save_path = os.path.join(MODEL_DIR, 'optimizer.pth')
torch.save(optimizer.state_dict(), optimizer_save_path)
print(f"Optimizer state_dict saved to {optimizer_save_path}")

scheduler_save_path = os.path.join(MODEL_DIR, 'scheduler.pth')
torch.save(scheduler.state_dict(), scheduler_save_path)
print(f"Scheduler state_dict saved to {scheduler_save_path}")

# ---------------------------
# 14. Define Evaluation Function
# ---------------------------

def detailed_evaluation(model, loader, device):
    """
    Generates detailed classification reports for each task.
    """
    model.eval()
    all_labels = {'hate': [], 'offensive': [], 'sentiment': []}
    all_preds = {'hate': [], 'offensive': [], 'sentiment': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tasks = batch['task']
            labels = batch['labels'].to(device)
            
            tasks_unique = set(tasks)
            for task in tasks_unique:
                idx = [i for i, t in enumerate(tasks) if t == task]
                if not idx:
                    continue
                task_input_ids = input_ids[idx]
                task_attention_mask = attention_mask[idx]
                task_labels = labels[idx]
                
                # Forward pass
                logits = model(input_ids=task_input_ids, attention_mask=task_attention_mask, task=task)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                # Collect labels and predictions
                all_preds[task].extend(preds)
                all_labels[task].extend(task_labels.cpu().numpy())
    
    # Generate classification reports
    for task in all_labels:
        print(f'\nClassification Report for {task.capitalize()}:')
        if task == 'hate':
            target_names = ['NOT', 'HATE']
        elif task == 'offensive':
            target_names = ['NOT', 'OFFN']
        elif task == 'sentiment':
            target_names = ['Negative', 'Positive']
        print(classification_report(all_labels[task], all_preds[task], target_names=target_names))
        print('-' * 50)

# ---------------------------
# 15. Load the Saved Model for Inference
# ---------------------------

# Load the saved model
model_load_path = final_model_path
if isinstance(model, nn.DataParallel):
    state_dict = torch.load(model_load_path, map_location=device)
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(torch.load(model_load_path, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from {model_load_path}")

# Detailed evaluation on test set
detailed_evaluation(model, test_loader, device)

# ---------------------------
# 16. Inference on Hindi Dataset
# ---------------------------

# Paths to Hindi datasets
hindi_tsv_paths = [
    os.path.join(HINDI_DATA_DIR, 'hasoc2019_hi_test_gold_2919.tsv'),
    os.path.join(HINDI_DATA_DIR, 'hindi_dataset.tsv')
]

# Load and combine Hindi datasets
hindi_dfs = []
for path in hindi_tsv_paths:
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t')
        hindi_dfs.append(df)
        print(f"Loaded Hindi dataset from {path} with shape {df.shape}")
    else:
        print(f"File not found: {path}")

if hindi_dfs:
    hindi_combined_df = pd.concat(hindi_dfs, ignore_index=True)
else:
    hindi_combined_df = pd.DataFrame(columns=['text', 'task_1'])
    print("No Hindi datasets loaded.")

# Map labels
hindi_combined_df['label'] = hindi_combined_df['task_1'].map({'NOT': 'NOT', 'HOF': 'HATE'})
hindi_combined_df['clean_text'] = hindi_combined_df['text'].apply(clean_text)
hindi_combined_df = hindi_combined_df[['text', 'label', 'clean_text']]
print(f"\nCombined Hindi Dataset Shape: {hindi_combined_df.shape}")
print("\nCombined Hindi Dataset Sample:")
print(hindi_combined_df.head())

# Define Inference Dataset Class
class InferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe['clean_text'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': self.texts[idx]
        }

# Instantiate Inference Dataset and DataLoader
inference_dataset = InferenceDataset(hindi_combined_df, tokenizer, MAX_LEN)
inference_loader = DataLoader(
    inference_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

def predict(model, loader, device):
    """
    Runs inference on the data_loader and returns predictions for each task.
    """
    model.eval()
    predictions = {'hate': [], 'offensive': [], 'sentiment': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            for task in ['hate', 'offensive', 'sentiment']:
                logits = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions[task].extend(preds)
    return predictions

# Run inference
preds = predict(model, inference_loader, device)

# Map predictions to labels
hate_map = {0: 'NOT', 1: 'HATE'}
offensive_map = {0: 'NOT', 1: 'OFFN'}
sentiment_map = {0: 'Negative', 1: 'Positive'}

hindi_combined_df['hate_prediction'] = [hate_map.get(p, 'NOT') for p in preds['hate']]
hindi_combined_df['offensive_prediction'] = [offensive_map.get(p, 'NOT') for p in preds['offensive']]
hindi_combined_df['sentiment_prediction'] = [sentiment_map.get(p, 'Positive') for p in preds['sentiment']]

# Display sample predictions
print("\nSample Predictions:")
for i in range(min(5, len(hindi_combined_df))):
    print(f"Text: {hindi_combined_df['text'].iloc[i]}")
    print(f"True Label: {hindi_combined_df['label'].iloc[i]}")
    print(f"Hate Prediction: {hindi_combined_df['hate_prediction'].iloc[i]}")
    print(f"Offensive Prediction: {hindi_combined_df['offensive_prediction'].iloc[i]}")
    print(f"Sentiment Prediction: {hindi_combined_df['sentiment_prediction'].iloc[i]}")
    print("-" * 50)

# Classification Report for Hate Task
if 'hate' in preds and hindi_combined_df['label'].nunique() > 1:
    print("\nClassification Report for Hate Task:")
    print(classification_report(
        hindi_combined_df['label'],
        hindi_combined_df['hate_prediction'],
        target_names=['NOT', 'HATE']
    ))
else:
    print("\nNo 'hate' task predictions available or insufficient label diversity.")
