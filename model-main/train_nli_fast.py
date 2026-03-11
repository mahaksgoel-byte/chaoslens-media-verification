#!/usr/bin/env python3
"""
Fast NLI Model Training - Quick Improvement to 85%+ accuracy
Uses smaller model and optimized settings for speed
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

class FastNLIDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        premise, hypothesis, label = self.data[idx]
        
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        label_id = self.label_map[label]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

def load_nli_data_fast(filepath, max_samples=5000):
    """Load smaller dataset for faster training"""
    print(f"Loading NLI data from {filepath}...")
    
    data = []
    label_counts = {'contradiction': 0, 'neutral': 0, 'entailment': 0}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:max_samples]
    
    for line in tqdm(lines, desc="Processing data"):
        parts = line.strip().split('\t')
        if len(parts) == 3:
            premise, hypothesis, label = parts
            if label in label_counts:
                data.append((premise, hypothesis, label))
                label_counts[label] += 1
    
    print(f"Loaded {len(data)} samples")
    print(f"Label distribution: {label_counts}")
    
    # Use only contradiction and entailment (most common)
    filtered_data = [(p, h, l) for p, h, l in data if l in ['contradiction', 'entailment']]
    
    # Balance the dataset
    contradiction_data = [(p, h, l) for p, h, l in filtered_data if l == 'contradiction']
    entailment_data = [(p, h, l) for p, h, l in filtered_data if l == 'entailment']
    
    min_count = min(len(contradiction_data), len(entailment_data))
    balanced_data = contradiction_data[:min_count] + entailment_data[:min_count]
    
    print(f"Balanced dataset: {len(balanced_data)} samples")
    print(f"Contradiction: {len(contradiction_data[:min_count])}, Entailment: {len(entailment_data[:min_count])}")
    
    label_map = {'contradiction': 0, 'entailment': 1}
    return balanced_data, label_map

def train_nli_fast():
    """Fast NLI training with optimized settings"""
    
    # Fast configuration
    config = {
        'model_name': 'distilbert-base-uncased',  # Much faster than BART
        'max_length': 128,  # Reduced for speed
        'batch_size': 32,   # Larger batch for efficiency
        'learning_rate': 3e-5,
        'epochs': 3,        # Fewer epochs
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'seed': 42
    }
    
    print("FAST NLI TRAINING - Target: 85%+ accuracy")
    print("=" * 50)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Load data
    data, label_map = load_nli_data_fast('/Users/mehakgoel/Desktop/Bluebit/model-main/training_data/nli_training.txt')
    
    # Split data
    train_data, val_data = train_test_split(
        data, test_size=0.15, random_state=config['seed']
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Load tokenizer and model
    print("Loading DistilBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'], 
        num_labels=2  # Only 2 classes: contradiction, entailment
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = FastNLIDataset(train_data, tokenizer, label_map, config['max_length'])
    val_dataset = FastNLIDataset(val_data, tokenizer, label_map, config['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=total_steps
    )
    
    # Training loop
    best_accuracy = 0
    
    print("\nStarting fast training...")
    print("=" * 40)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 30)
        
        # Training
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_train_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1)
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        
        # Validation
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                preds = torch.argmax(outputs.logits, dim=1)
                val_predictions.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_labels, val_predictions)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            output_dir = "/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_nli_fast"
            os.makedirs(output_dir, exist_ok=True)
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save mappings
            with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
                json.dump(label_map, f)
            
            print(f"✓ New best model saved: {val_accuracy*100:.2f}%")
    
    print(f"\nFast training completed!")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    
    return best_accuracy

if __name__ == "__main__":
    accuracy = train_nli_fast()
    print(f"\nFinal Fast NLI Accuracy: {accuracy*100:.2f}%")
