#!/usr/bin/env python3
"""
Fast Intent Classifier Training Script
Target: Quick training with good accuracy
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
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

class FastIntentDataset(Dataset):
    def __init__(self, data, tokenizer, intent_to_id, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_to_id = intent_to_id
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, intent = self.data[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        label_id = self.intent_to_id[intent]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

class FastIntentClassifier(nn.Module):
    def __init__(self, base_model, num_classes, hidden_size=768, dropout=0.2):
        super(FastIntentClassifier, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token from last hidden state
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]  # [CLS] token
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits

def load_intent_data_fast(filepath, max_samples=3000):
    """Load smaller dataset for faster training"""
    print(f"Loading intent data from {filepath}...")
    
    data = []
    intent_counts = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:max_samples]
    
    for line in tqdm(lines, desc="Processing data"):
        parts = line.strip().split('\t')
        if len(parts) == 2:
            text, intent = parts
            data.append((text, intent))
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"Loaded {len(data)} samples")
    print(f"Intent distribution: {intent_counts}")
    
    # Focus on main classes with enough samples
    main_intents = {k: v for k, v in intent_counts.items() if v >= 50}
    
    if len(main_intents) < 2:
        print("Warning: Not enough samples per class. Using all available data.")
        intent_to_id = {intent: idx for idx, intent in enumerate(sorted(intent_counts.keys()))}
        return data, intent_to_id
    
    # Filter data to main intents
    filtered_data = [(text, intent) for text, intent in data if intent in main_intents]
    
    print(f"Filtered to {len(filtered_data)} samples with main intents")
    print(f"Main intents: {list(main_intents.keys())}")
    
    # Create intent mapping
    intent_to_id = {intent: idx for idx, intent in enumerate(sorted(main_intents.keys()))}
    
    return filtered_data, intent_to_id

def train_intent_fast():
    """Fast intent classifier training"""
    
    # Fast configuration
    config = {
        'model_name': 'distilbert-base-uncased',
        'max_length': 64,    # Shorter sequences for speed
        'batch_size': 32,    # Larger batch
        'learning_rate': 5e-5,
        'epochs': 3,         # Fewer epochs
        'warmup_steps': 50,
        'weight_decay': 0.01,
        'seed': 42
    }
    
    print("FAST INTENT TRAINING")
    print("=" * 40)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Load data
    data, intent_to_id = load_intent_data_fast('/Users/mehakgoel/Desktop/Bluebit/model-main/training_data/intent_training.txt')
    
    if len(data) < 100:
        print("Not enough data for training")
        return 0
    
    # Create reverse mapping
    id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}
    
    print(f"Intent mapping: {intent_to_id}")
    
    # Split data
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=config['seed']
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Load tokenizer and base model
    print("Loading DistilBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    base_model = AutoModel.from_pretrained(config['model_name'])
    
    # Create custom classifier
    model = FastIntentClassifier(base_model, len(intent_to_id))
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = FastIntentDataset(train_data, tokenizer, intent_to_id, config['max_length'])
    val_dataset = FastIntentDataset(val_data, tokenizer, intent_to_id, config['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0
    
    print("\nStarting fast training...")
    print("=" * 35)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 25)
        
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            total_train_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
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
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                preds = torch.argmax(outputs, dim=1)
                val_predictions.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_labels, val_predictions)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            output_dir = "/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_intent_fast"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'intent_to_id': intent_to_id,
                'id_to_intent': id_to_intent,
                'intents': list(intent_to_id.keys())
            }, os.path.join(output_dir, 'intent_model.pth'))
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            
            # Save mappings
            mappings = {
                'intent_to_id': intent_to_id,
                'id_to_intent': id_to_intent,
                'intents': list(intent_to_id.keys()),
                'config': config
            }
            with open(os.path.join(output_dir, 'intent_mappings.json'), 'w') as f:
                json.dump(mappings, f, indent=2)
            
            print(f"✓ New best model saved: {val_accuracy*100:.2f}%")
    
    print(f"\nFast training completed!")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    
    return best_accuracy

if __name__ == "__main__":
    accuracy = train_intent_fast()
    print(f"\nFinal Fast Intent Accuracy: {accuracy*100:.2f}%")
