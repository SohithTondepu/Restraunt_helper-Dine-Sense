import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'Restaurant reviews.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'Restaurant reviews.csv')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'deberta_finetuned.pt')

def train_and_save_model():
    print("=== STARTING PYTORCH FINE-TUNING ON RESTAURANT REVIEWS ===")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load & Clean Dataset
    df = pd.read_csv(DATA_PATH)
    df = df[df['Rating'] != 'Like'].copy()
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df.dropna(subset=['Rating', 'Review'], inplace=True)
    
    # Label Mapping: 0 = Negative (<= 2.5), 1 = Neutral (3.0, 3.5), 2 = Positive (>= 4.0)
    def map_sentiment(r):
        if r <= 2.5:
            return 0
        elif r <= 3.5:
            return 1
        else:
            return 2
            
    df['label'] = df['Rating'].apply(map_sentiment)
    
    reviews = df['Review'].astype(str).tolist()
    labels = df['label'].tolist()
    
    print(f"Total Cleaned Reviews for Fine-Tuning: {len(reviews)}")
    
    # 2. Tokenize using Transformer Tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    encodings = tokenizer(
        reviews,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Load Base Transformer Model with Classification Head (3 Classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing fine-tuning on device: {device}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 1
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # 4. PyTorch Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
                print(f"Epoch {epoch} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch} Complete | Average Loss: {avg_loss:.4f} ===")
        
    # 5. Save Fine-Tuned PyTorch Model Checkpoint
    print(f"Saving fine-tuned PyTorch model checkpoint to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_DIR)
    print("=== MODEL FINE-TUNING AND PERSISTENCE COMPLETE ===")

if __name__ == '__main__':
    train_and_save_model()
