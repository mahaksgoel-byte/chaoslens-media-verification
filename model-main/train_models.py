#!/usr/bin/env python3
"""
Simple training script for all models using FEVER dataset
"""

import os
import torch
import random
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer,
    AdamW, get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm

# Training configuration
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
TRAIN_SIZE = 50000  # Use 50k samples as requested

class NLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        premise, hypothesis, label = self.data[idx]
        
        inputs = self.tokenizer(
            premise, hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert label to integer
        label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        label_id = label_map[label]
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

class T5Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

def load_nli_data(filepath, max_samples=TRAIN_SIZE):
    """Load NLI training data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:max_samples]
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            premise, hypothesis, label = parts
            data.append((premise, hypothesis, label))
    
    return data

def load_splitter_data(filepath, max_samples=TRAIN_SIZE):
    """Load T5 splitter training data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:max_samples]
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            input_text, target_text = parts
            data.append((input_text, target_text))
    
    return data

def load_intent_data(filepath, max_samples=TRAIN_SIZE):
    """Load intent classifier training data"""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:max_samples]
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            text, intent = parts
            examples.append(InputExample(texts=[text], label=intent))
    
    return examples

def train_nli_model():
    """Train the NLI verifier model"""
    print("Training NLI model...")
    
    # Load data
    train_data = load_nli_data('/Users/mehakgoel/Desktop/Bluebit/model-main/training_data/nli_training.txt')
    print(f"Loaded {len(train_data)} NLI training samples")
    
    # Initialize model and tokenizer
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Create dataset and dataloader
    dataset = NLIDataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"NLI Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"NLI Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    output_dir = "/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_nli"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"NLI model saved to {output_dir}")

def train_splitter_model():
    """Train the T5 claim splitter model"""
    print("Training T5 claim splitter model...")
    
    # Load data
    train_data = load_splitter_data('/Users/mehakgoel/Desktop/Bluebit/model-main/training_data/splitter_training.txt')
    print(f"Loaded {len(train_data)} splitter training samples")
    
    # Initialize model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Create dataset and dataloader
    dataset = T5Dataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Splitter Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Splitter Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    output_dir = "/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_splitter"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Splitter model saved to {output_dir}")

def train_intent_model():
    """Train the intent classifier model"""
    print("Training intent classifier model...")
    
    # Load data
    train_examples = load_intent_data('/Users/mehakgoel/Desktop/Bluebit/model-main/training_data/intent_training.txt')
    print(f"Loaded {len(train_examples)} intent training samples")
    
    # Initialize model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    # Create training data
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=100,
        output_path="/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_intent"
    )
    print(f"Intent model saved to /Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_intent")

def main():
    """Train all models"""
    print("Starting training on FEVER dataset...")
    print(f"Training samples: {TRAIN_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 50)
    
    # Train NLI model
    train_nli_model()
    print("-" * 50)
    
    # Train T5 splitter model
    train_splitter_model()
    print("-" * 50)
    
    # Train intent classifier
    train_intent_model()
    print("-" * 50)
    
    print("All models trained successfully!")

if __name__ == "__main__":
    main()
