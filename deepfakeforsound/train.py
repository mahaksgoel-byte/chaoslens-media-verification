"""
Training script for multi-modal deepfake audio detection
Implements staged training with contrastive learning
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Suppress torchcodec loading to avoid DLL issues
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''

# COMPLETELY DISABLE TORCHCODEC
os.environ['TORCH_AUDIO_DISABLE_TORCHCODEC'] = '1'
os.environ['LIBROKA_DISABLE_TORCHCODEC'] = '1'

# Suppress all warnings for clean output
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.cuda.amp import GradScaler, autocast  # Removed for CPU-only
from torch.utils.tensorboard import SummaryWriter

import yaml
from tqdm import tqdm
import numpy as np

from utils import (
    set_seed, setup_logging, get_device, load_config, ensure_dir,
    save_checkpoint, load_checkpoint, AverageMeter, EarlyStopping,
    count_parameters, compute_eer
)
from datasets import create_dataloaders
from features import AudioFeatureExtractor, EMGFeatureExtractor
from models import create_model
from augmentation import AudioAugmentation


class WarmupScheduler:
    """
    Learning rate scheduler with warmup
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        scheduler_type: str = "cosine"
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.scheduler_type = scheduler_type
        
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            if self.scheduler_type == "cosine":
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            else:
                lr = self.base_lr * (1 - progress)
        
        lr = max(lr, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_last_lr(self) -> list:
        """Get last learning rate as list for compatibility"""
        return [self.get_lr()]


class Trainer:
    """
    Multi-modal deepfake detection trainer
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device(config['device'])
        
        # Setup logging
        self.logger = setup_logging(
            config['logging']['level'],
            log_file=os.path.join(config.get('output_dir', 'logs'), 'training.log')
        )
        
        # Set seeds
        set_seed(config['seed'], config['deterministic'])
        
        # Create output directories
        self.output_dir = ensure_dir(config.get('output_dir', 'outputs'))
        self.checkpoint_dir = ensure_dir(os.path.join(self.output_dir, 'checkpoints'))
        self.log_dir = ensure_dir(os.path.join(self.output_dir, 'logs'))
        
        # Initialize components
        self._setup_components()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stopping = EarlyStopping(
            patience=1000000,
            min_delta=0.001
        )
        
        # TensorBoard
        if config['logging']['tensorboard']:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
        
        # Mixed precision
        self.use_amp = False  # Force disable for CPU
        self.scaler = None
        
        self.logger.info("Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_components(self):
        """Setup model, data, and optimizers"""
        # Feature extractors
        self.audio_extractor = AudioFeatureExtractor(self.config).to(self.device)
        self.emg_extractor = EMGFeatureExtractor(self.config).to(self.device)
        
        # DISABLE AUGMENTATION TO FIX 4D ISSUES
        # augmented = self.augmentation(audio)
        self.audio_augmentation = None  # Skip augmentation
        
        # Model
        self.model = create_model(self.config).to(self.device)
        
        # Compile model if available
        if self.config.get('compile', False) and hasattr(torch, 'compile'):
            self.logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)
        
        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.config, transform=self.audio_augmentation
        )
        
        # Optimizer
        optimizer_config = self.config['training']['optimizer']
        # Ensure weight_decay is a float
        weight_decay = float(optimizer_config['weight_decay'])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=weight_decay,
            betas=optimizer_config['betas']
        )
        
        # Scheduler
        scheduler_config = self.config['training']['scheduler']
        self.scheduler = WarmupScheduler(
            self.optimizer,
            warmup_epochs=scheduler_config['warmup_epochs'],
            total_epochs=self.config['training']['epochs'],
            min_lr=scheduler_config['min_lr'],
            scheduler_type=scheduler_config['type']
        )
        
        # Log model info
        model_info = self.model.get_model_info()
        self.logger.info(f"Model parameters: {model_info['total_parameters']:,}")
        self.logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Resume training if specified
        resume_path = self.config['training']['checkpoint'].get('resume')
        if resume_path:
            self._load_checkpoint(resume_path)

        # Ensure internal epoch state matches resumed checkpoint (if any)
    
    def _get_current_stage(self) -> Dict[str, Any]:
        """Get current training stage configuration"""
        epoch = self.current_epoch
        stages = self.config['training']['stages']
        
        # Handle null stages (no stage training)
        if stages is None:
            return {
                'freeze_emg': False,
                'contrastive': False
            }
        
        for stage_name, stage_config in stages.items():
            if stage_config['start'] <= epoch < stage_config['end']:
                return stage_config

        # Default: pick the stage with the largest 'end' (i.e., last stage)
        try:
            last_stage = max(stages.values(), key=lambda s: s.get('end', 0))
            return last_stage
        except Exception:
            # Fallback: return default stage config
            return {
                'freeze_emg': False,
                'contrastive': False
            }
    
    def _apply_stage_configuration(self, stage_config: Dict[str, Any]):
        """Apply stage-specific configuration"""
        # Freeze/unfreeze branches
        if stage_config.get('freeze_emg', False):
            self.model.freeze_emg_branch()
        else:
            self.model.unfreeze_emg_branch()
        
        # Enable/disable contrastive learning
        if stage_config.get('contrastive', False):
            self.model.enable_contrastive()
        else:
            self.model.disable_contrastive()
        
        self.logger.info(f"Applied stage configuration: {stage_config}")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with better regularization"""
        self.model.train()
        
        # Get current stage configuration
        stage_config = self._get_current_stage()
        
        # Apply stage configuration
        self._apply_stage_configuration(stage_config)
        
        # Metrics
        losses = AverageMeter()
        classification_losses = AverageMeter()
        contrastive_losses = AverageMeter()
        accuracies = AverageMeter()
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, batch in enumerate(pbar):
            # Limit to 100 batches per epoch for speed
            if batch_idx >= 100:
                break
            # Validate batch is not empty
            if batch is None or len(batch) == 0:
                print(f"⚠️ Empty batch {batch_idx}, skipping")
                continue
            
            # Check if batch has valid audio
            if 'audio' not in batch or batch['audio'].numel() == 0:
                print(f"⚠️ No valid audio in batch {batch_idx}, skipping")
                continue
            
            # Get data
            audio = batch['audio'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Handle EMG if available
            if 'emg' in batch:
                emg_features = batch['emg'].to(self.device)
            else:
                emg_features = None
            
            self.optimizer.zero_grad()
            
            # Forward pass
            # Enable mixed precision for speed
            use_amp = self.config['training'].get('mixed_precision', False)
            if use_amp:
                from torch.cuda.amp import autocast
                # Note: CPU doesn't support autocast, but we'll keep it for GPU compatibility
                pass
            
            # FIX: Ensure audio is 2D [batch_size, samples]
            if audio.dim() == 3 and audio.shape[1] == 1:
                audio = audio.squeeze(1)  # Remove channel dimension
            elif audio.dim() > 3:
                audio = audio.reshape(audio.shape[0], -1)  # Flatten extra dimensions
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0)  # Add batch dimension
            
            # Ensure 2D shape [batch, samples]
            if audio.dim() != 2:
                audio = audio.reshape(1, -1)  # Force to [1, samples]
            
            # Ensure correct length (with safe config access)
            sample_rate = self.config.get('data', {}).get('sample_rate', 16000)
            duration = self.config.get('data', {}).get('duration', 4.0)
            target_length = int(sample_rate * duration)
            
            if audio.shape[1] < target_length:
                # Pad if too short
                padding = target_length - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            elif audio.shape[1] > target_length:
                # Truncate if too long
                audio = audio[:, :target_length]
            
            # Extract features with error handling
            try:
                with torch.no_grad():
                    audio_features_dict = self.audio_extractor(audio)
                    audio_features = self.audio_extractor.get_feature_tensor(audio_features_dict)
            except Exception as e:
                print(f"⚠️ Validation feature extraction failed: {e}")
                # Create dummy features
                audio_features = torch.zeros(len(audio), 128, 128, 128)  # Dummy features
            
            emg_features = None
            if 'emg' in batch and batch['emg'] is not None:
                try:
                    emg_data = batch['emg']
                    emg_features_dict = self.emg_extractor(emg_data)
                    emg_features = self.emg_extractor.get_feature_tensor(emg_features_dict)
                except Exception as e:
                    print(f"⚠️ Validation EMG feature extraction failed: {e}")
                    emg_features = None
            
            # Model forward with error handling
            try:
                outputs = self.model(audio_features, emg_features, return_embeddings=False)
                # Extract logits if model returns dict
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('probabilities', list(outputs.values())[0]))
                else:
                    logits = outputs
            except Exception as e:
                print(f"⚠️ Model forward failed batch {batch_idx}: {e}")
                continue  # Skip this batch
            
            # Apply label smoothing to prevent overconfidence
            label_smoothing = self.config['training']['loss'].get('label_smoothing', 0.0)
            smoothed_labels = labels * (1.0 - label_smoothing) + (label_smoothing / 2.0)
            
            # Compute loss with stronger regularization
            contrastive_weight = (
                self.config['model']['contrastive']['weight'] 
                if stage_config.get('contrastive', False) else 0.0
            )
            loss_dict = self.model.compute_loss(outputs, smoothed_labels, contrastive_weight)
            
            total_loss = loss_dict['total_loss']
            
            # Add L2 regularization - FIX: Properly convert to Python float
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.model.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)
            
            # Get weight_decay as float (handles both list and scalar cases)
            weight_decay_config = self.config['training']['regularization']['weight_decay']
            if isinstance(weight_decay_config, (list, tuple)):
                weight_decay = float(weight_decay_config[0])  # Extract from list/tuple
            else:
                weight_decay = float(weight_decay_config)  # Convert to float
            
            l2_loss = weight_decay * l2_reg.item()  # Convert tensor to float
            total_loss = total_loss + l2_loss
            
            # Backward pass with NaN/Inf checking
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ NaN/Inf loss detected in batch {batch_idx}, skipping batch")
                continue
            
            total_loss.backward()
            
            # Gradient clipping
            if self.config['training']['regularization']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['regularization']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            losses.update(total_loss.item(), audio.size(0))
            classification_losses.update(loss_dict['classification_loss'].item(), audio.size(0))
            
            if 'contrastive_loss' in loss_dict:
                contrastive_losses.update(loss_dict['contrastive_loss'].item(), audio.size(0))
            
            # Accuracy
            with torch.no_grad():
                # Get probabilities from the correct source
                if isinstance(outputs, dict):
                    probabilities = outputs.get('probabilities', outputs.get('logits', logits))
                else:
                    probabilities = outputs
                
                predictions = (probabilities > 0.5).float().squeeze()
                
                # Handle scalar predictions
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Ensure predictions and labels have same size
                if len(predictions.shape) > 0 and len(labels.shape) > 0:
                    if predictions.shape[0] != labels.shape[0]:
                        min_size = min(predictions.shape[0], labels.shape[0])
                        predictions = predictions[:min_size]
                        labels = labels[:min_size]
                    
                    accuracy = (predictions == labels).float().mean().item()
                    accuracies.update(accuracy, predictions.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Acc': f"{accuracies.avg:.3f}",
                'Loss': f"{losses.avg:.4f}",
                'LR': f"{self.scheduler.get_lr():.6f}"
            })
        
        # Log epoch metrics
        epoch_metrics = {
            'epoch': self.current_epoch,
            'train_loss': losses.avg,
            'train_classification_loss': classification_losses.avg,
            'train_contrastive_loss': contrastive_losses.avg if contrastive_losses.count > 0 else 0.0,
            'train_accuracy': accuracies.avg,
            'learning_rate': self.scheduler.get_lr()
        }
        
        # Print accuracy after each epoch
        print(f"\nEpoch {self.current_epoch}/20 - Accuracy: {accuracies.avg:.3f} ({accuracies.avg*100:.1f}%)")
        print(f"Loss: {losses.avg:.4f} - LR: {self.scheduler.get_lr():.6f}")
        
        return epoch_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        # Metrics
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                # Move data to device
                audio = batch['audio'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # FIX: Ensure audio is 2D [batch_size, samples]
                if audio.dim() == 3 and audio.shape[1] == 1:
                    audio = audio.squeeze(1)  # Remove channel dimension
                elif audio.dim() > 3:
                    audio = audio.reshape(audio.shape[0], -1)  # Flatten extra dimensions
                elif audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # Add batch dimension
                
                # Ensure 2D shape [batch, samples]
                if audio.dim() != 2:
                    audio = audio.reshape(1, -1)  # Force to [1, samples]
                
                # Extract features with error handling
                try:
                    with torch.no_grad():
                        audio_features_dict = self.audio_extractor(audio)
                        audio_features = self.audio_extractor.get_feature_tensor(audio_features_dict)
                except Exception as e:
                    print(f"⚠️ Validation feature extraction failed: {e}")
                    # Create dummy features
                    audio_features = torch.zeros(len(audio), 128, 128, 128)  # Dummy features
                
                emg_features = None
                if 'emg' in batch and batch['emg'] is not None:
                    try:
                        emg_data = batch['emg']
                        emg_features_dict = self.emg_extractor(emg_data)
                        emg_features = self.emg_extractor.get_feature_tensor(emg_features_dict)
                    except Exception as e:
                        print(f"⚠️ Validation EMG feature extraction failed: {e}")
                        emg_features = None
                
                # Model forward with error handling
                try:
                    outputs = self.model(audio_features, emg_features, return_embeddings=False)
                    # Extract logits if model returns dict
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('probabilities', list(outputs.values())[0]))
                    else:
                        logits = outputs
                except Exception as e:
                    print(f"⚠️ Validation model forward failed: {e}")
                    continue  # Skip this batch
                
                # Compute loss (without contrastive)
                loss_dict = self.model.compute_loss(outputs, labels, contrastive_weight=0.0)
                total_loss = loss_dict['total_loss']
                
                # Update metrics
                losses.update(total_loss.item(), audio.size(0))
                
                # Accuracy
                predictions = outputs['probabilities'].squeeze()
                accuracy = ((predictions > 0.5).float() == labels).float().mean().item()
                accuracies.update(accuracy, audio.size(0))
                
                # Store for detailed metrics
                if predictions.dim() > 0:
                    all_predictions.extend(predictions.cpu().numpy())
                else:
                    all_predictions.append(predictions.cpu().numpy().item())
                if labels.dim() > 0:
                    all_labels.extend(labels.cpu().numpy())
                else:
                    all_labels.append(labels.cpu().numpy().item())
        
        # Compute detailed metrics only if we have predictions
        val_metrics = {
            'val_loss': losses.avg,
            'val_accuracy': accuracies.avg,
            'val_auc': 0.0,
            'val_eer': 0.0
        }
        
        if len(all_predictions) > 0 and len(all_labels) > 0:
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            # AUC
            try:
                from sklearn.metrics import roc_auc_score
                val_metrics['val_auc'] = roc_auc_score(all_labels, all_predictions)
            except Exception as e:
                print(f"⚠️ AUC computation failed: {e}")
            
            # EER
            try:
                eer, _ = compute_eer(all_predictions, all_labels)
                val_metrics['val_eer'] = eer
            except Exception as e:
                print(f"⚠️ EER computation failed: {e}")
        
        # Print validation accuracy
        print(f"Validation Accuracy: {accuracies.avg:.3f} ({accuracies.avg*100:.1f}%)")
        
        return val_metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.__dict__.copy(),
                'metrics': metrics,
                'config': self.config
            }
            
            # Try to save regular checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            try:
                save_checkpoint(checkpoint, self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt', is_best)
                print(f"✅ Checkpoint saved for epoch {epoch}")
            except Exception as save_error:
                print(f"⚠️ Could not save checkpoint file: {save_error}")
                # Try alternative save location
                alt_path = f'checkpoint_epoch_{epoch}.pt'
                torch.save(checkpoint, alt_path)
                print(f"✅ Alternative checkpoint saved: {alt_path}")
            
            # Try to save latest
            try:
                latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
                save_checkpoint(checkpoint, self.checkpoint_dir, 'latest.pt')
            except Exception as latest_error:
                print(f"⚠️ Could not save latest: {latest_error}")
                # Alternative latest
                torch.save(checkpoint, 'latest.pt')
                print("✅ Alternative latest saved: latest.pt")
            
        except Exception as e:
            print(f"❌ Checkpoint saving failed (but training continues): {e}")
            # Don't stop training, just continue
            pass
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device
        )

        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

        self.logger.info(
            f"Resumed from epoch={checkpoint.get('epoch', 0)} -> next_epoch={self.current_epoch}, "
            f"best_val_acc={self.best_val_acc:.4f}"
        )

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        epochs = self.config['training']['epochs']
        save_freq = self.config['training']['checkpoint']['save_freq']
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Validate epoch
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()
                metrics = {**train_metrics, **val_metrics}
            else:
                metrics = train_metrics
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            if self.writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(key, value, epoch)
            
            # Log to console
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {metrics['train_loss']:.4f}, "
                f"Train Acc: {metrics['train_accuracy']:.4f}, "
                f"Val Loss: {metrics.get('val_loss', 0):.4f}, "
                f"Val Acc: {metrics.get('val_accuracy', 0):.4f}, "
                f"Val AUC: {metrics.get('val_auc', 0):.4f}, "
                f"Val EER: {metrics.get('val_eer', 0):.4f}"
            )
            
            # Check if best model
            is_best = False
            if 'val_accuracy' in metrics and metrics['val_accuracy'] > self.best_val_acc:
                self.best_val_acc = metrics['val_accuracy']
                is_best = True
                self.logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}")
            
            # Save checkpoint
            if epoch % save_freq == 0 or is_best:
                self._save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping
            if 'val_accuracy' in metrics:
                if self.early_stopping(metrics['val_accuracy']):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        return self.best_val_acc


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multi-modal deepfake detector')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with args
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.resume:
        config['training']['checkpoint']['resume'] = args.resume
    if args.device:
        config['device'] = args.device
    
    # Create trainer and start training
    trainer = Trainer(config)
    best_accuracy = trainer.train()
    
    print(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()