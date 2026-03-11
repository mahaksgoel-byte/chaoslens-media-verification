"""
Multi-modal fusion model combining audio and EMG branches
Cross-modal attention and contrastive learning for deepfake detection
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_branch import AudioBranch, create_audio_branch
from .emg_branch import EMGBranch, create_emg_branch


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for audio-EMG fusion
    """
    
    def __init__(
        self,
        audio_dim: int,
        emg_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.emg_dim = emg_dim
        self.n_heads = n_heads
        self.d_model = max(audio_dim, emg_dim)
        
        # Projections to common dimension
        self.audio_proj = nn.Linear(audio_dim, self.d_model)
        self.emg_proj = nn.Linear(emg_dim, self.d_model)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projections
        self.audio_out = nn.Linear(self.d_model, audio_dim)
        self.emg_out = nn.Linear(self.d_model, emg_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_audio = nn.LayerNorm(audio_dim)
        self.layer_norm_emg = nn.LayerNorm(emg_dim)
    
    def forward(self, audio_emb: torch.Tensor, emg_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention
        
        Args:
            audio_emb: Audio embedding [batch_size, audio_dim]
            emg_emb: EMG embedding [batch_size, emg_dim]
            
        Returns:
            Enhanced audio and EMG embeddings
        """
        batch_size = audio_emb.shape[0]
        
        # Project to common dimension
        audio_proj = self.audio_proj(audio_emb)  # [batch, d_model]
        emg_proj = self.emg_proj(emg_emb)  # [batch, d_model]
        
        # Stack for attention [batch, 2, d_model]
        combined = torch.stack([audio_proj, emg_proj], dim=1)
        
        # Apply self-attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Split attended features
        audio_attended = attended[:, 0, :]  # [batch, d_model]
        emg_attended = attended[:, 1, :]  # [batch, d_model]
        
        # Project back to original dimensions
        audio_enhanced = self.audio_out(audio_attended)
        emg_enhanced = self.emg_out(emg_attended)
        
        # Residual connections
        audio_enhanced = self.layer_norm_audio(audio_emb + self.dropout(audio_enhanced))
        emg_enhanced = self.layer_norm_emg(emg_emb + self.dropout(emg_enhanced))
        
        return audio_enhanced, emg_enhanced


class FusionLayer(nn.Module):
    """
    Fusion layer combining audio and EMG embeddings
    """
    
    def __init__(
        self,
        audio_dim: int,
        emg_dim: int,
        fusion_dim: int,
        dropout: float = 0.4
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.emg_dim = emg_dim
        self.fusion_dim = fusion_dim
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            audio_dim, emg_dim, n_heads=8, dropout=dropout
        )
        
        # Concatenation and projection
        combined_dim = audio_dim + emg_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, audio_emb: torch.Tensor, emg_emb: torch.Tensor) -> torch.Tensor:
        """
        Fuse audio and EMG embeddings
        
        Args:
            audio_emb: Audio embedding [batch_size, audio_dim]
            emg_emb: EMG embedding [batch_size, emg_dim]
            
        Returns:
            Fused embedding [batch_size, fusion_dim]
        """
        # Apply cross-modal attention
        audio_enhanced, emg_enhanced = self.cross_attention(audio_emb, emg_emb)
        
        # Concatenate enhanced embeddings
        combined = torch.cat([audio_enhanced, emg_enhanced], dim=1)  # [batch, audio_dim + emg_dim]
        
        # Project to fusion dimension
        fused = self.fusion_proj(combined)
        
        # Layer normalization
        fused = self.layer_norm(fused)
        
        return fused


class ContrastiveProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings for contrastive learning
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            Projected embeddings [batch_size, projection_dim]
        """
        # L2 normalize the output
        projections = self.projection_head(x)
        projections = F.normalize(projections, dim=1)
        
        return projections


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for contrastive learning
    """
    
    def __init__(self, temperature: float = 0.07, batch_size: int = 32):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.mask = self._create_mask()
    
    def _create_mask(self) -> torch.Tensor:
        """Create mask to avoid positive pairs with itself"""
        mask = torch.ones(self.batch_size, self.batch_size) - torch.eye(self.batch_size)
        return mask.bool()
    
    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss
        
        Args:
            projections: Projected embeddings [batch_size, projection_dim]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Contrastive loss
        """
        batch_size = projections.shape[0]
        
        # Adjust mask if batch size changed
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.mask = self._create_mask()
        
        device = projections.device
        self.mask = self.mask.to(device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create positive and negative masks
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.T).float() * self.mask.float()
        negative_mask = 1.0 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask = positive_mask - torch.eye(batch_size, device=device)
        
        # Compute loss
        # For each sample, compute loss with all positive and negative samples
        exp_sim = torch.exp(similarity_matrix)
        
        # Numerator: sum of positive similarities
        positive_sim = exp_sim * positive_mask
        numerator = torch.sum(positive_sim, dim=1)
        
        # Denominator: sum of all similarities (excluding self)
        all_sim = exp_sim * self.mask.float()
        denominator = torch.sum(all_sim, dim=1)
        
        # Compute loss
        loss = -torch.log(numerator / (denominator + 1e-8))
        
        # Only compute loss for samples that have positive pairs
        has_positive = torch.sum(positive_mask, dim=1) > 0
        loss = loss[has_positive]
        
        return torch.mean(loss) if len(loss) > 0 else torch.tensor(0.0, device=device)


class ClassificationHead(nn.Module):
    """
    Classification head for deepfake detection
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.4,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Build classifier layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Classification logits [batch_size, 1]
        """
        return self.classifier(x)
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss
        
        Args:
            logits: Model outputs [batch_size, 1]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Classification loss
        """
        # Ensure batch sizes match
        if logits.shape[0] != labels.shape[0]:
            min_batch_size = min(logits.shape[0], labels.shape[0])
            logits = logits[:min_batch_size]
            labels = labels[:min_batch_size]
        
        if self.use_focal:
            return self._focal_loss(logits, labels)
        else:
            return F.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1))
    
    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss"""
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        return focal_loss.mean()


class MultiModalDeepfakeDetector(nn.Module):
    """
    Complete multi-modal deepfake detection model
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        
        # Audio and EMG branches
        self.audio_branch = AudioBranch(config=config, input_channels=1)
        self.emg_branch = EMGBranch(config=config, input_channels=8)
        
        # Check if EMG is disabled
        emg_disabled = getattr(self.emg_branch, 'disabled', False)
        
        # Fusion layer - handle disabled EMG
        if emg_disabled or self.model_config.get('fusion') is None:
            # Simple audio-only classifier
            self.fusion_layer = None
            self.classifier = ClassificationHead(
                input_dim=self.model_config['audio_branch']['embedding_dim'],
                hidden_dims=[128, 64],
                dropout=0.2,
                use_focal=config['training']['loss']['focal'],
                focal_alpha=config['training']['loss']['focal_alpha'],
                focal_gamma=config['training']['loss']['focal_gamma']
            )
        else:
            # Full fusion model
            self.fusion_layer = FusionLayer(
                audio_dim=self.model_config['audio_branch']['embedding_dim'],
                emg_dim=self.model_config['emg_branch']['embedding_dim'],
                fusion_dim=self.model_config['fusion']['projection_dim'],
                dropout=self.model_config['fusion']['dropout']
            )
            
            self.classifier = ClassificationHead(
                input_dim=self.model_config['fusion']['projection_dim'],
                hidden_dims=[256, 128],
                dropout=self.model_config['fusion']['dropout'],
                use_focal=config['training']['loss']['focal'],
                focal_alpha=config['training']['loss']['focal_alpha'],
                focal_gamma=config['training']['loss']['focal_gamma']
            )
        
        # Contrastive learning components - disabled for fast training
        self.contrastive_enabled = False
        self.contrastive_head = None
        self.contrastive_loss = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        emg_features: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model
        
        Args:
            audio_features: Audio features [batch_size, channels, freq, time]
            emg_features: EMG features [batch_size, channels, time] (optional)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with outputs
        """
        outputs = {}
        
        # Audio branch
        audio_embedding = self.audio_branch(audio_features)
        outputs['audio_embedding'] = audio_embedding
        
        # Check if fusion is disabled (audio-only mode)
        if self.fusion_layer is None:
            # Audio-only mode
            fused_embedding = audio_embedding
        else:
            # Fusion mode
            # EMG branch (if available)
            if emg_features is not None:
                emg_embedding = self.emg_branch(emg_features)
                outputs['emg_embedding'] = emg_embedding
                
                # Fusion
                fused_embedding = self.fusion_layer(audio_embedding, emg_embedding)
            else:
                # Audio-only mode: create dummy EMG embedding
                emg_dim = self.model_config['emg_branch']['embedding_dim']
                emg_embedding = torch.zeros(audio_embedding.shape[0], emg_dim, device=audio_embedding.device)
                fused_embedding = self.fusion_layer(audio_embedding, emg_embedding)
        
        outputs['fused_embedding'] = fused_embedding
        
        # Classification
        logits = self.classifier(fused_embedding)
        outputs['logits'] = logits
        outputs['probabilities'] = torch.sigmoid(logits)
        
        # Contrastive projections (if enabled)
        if self.contrastive_enabled and self.contrastive_head is not None:
            contrastive_projections = self.contrastive_head(fused_embedding)
            outputs['contrastive_projections'] = contrastive_projections
        
        if return_embeddings:
            return outputs
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        contrastive_weight: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            contrastive_weight: Weight for contrastive loss
            
        Returns:
            Dictionary with losses
        """
        losses = {}
        
        # Classification loss
        classification_loss = self.classifier.compute_loss(outputs['logits'], labels)
        losses['classification_loss'] = classification_loss
        
        # Contrastive loss (if enabled)
        if (self.contrastive_enabled and 
            'contrastive_projections' in outputs and 
            contrastive_weight > 0):
            contrastive_loss = self.contrastive_loss(
                outputs['contrastive_projections'], labels
            )
            losses['contrastive_loss'] = contrastive_loss
            losses['total_loss'] = classification_loss + contrastive_weight * contrastive_loss
        else:
            losses['total_loss'] = classification_loss
        
        return losses
    
    def freeze_emg_branch(self):
        """Freeze EMG branch parameters"""
        for param in self.emg_branch.parameters():
            param.requires_grad = False
        logging.info("EMG branch frozen")
    
    def unfreeze_emg_branch(self):
        """Unfreeze EMG branch parameters"""
        for param in self.emg_branch.parameters():
            param.requires_grad = True
        logging.info("EMG branch unfrozen")
    
    def freeze_audio_branch(self):
        """Freeze audio branch parameters"""
        for param in self.audio_branch.parameters():
            param.requires_grad = False
        logging.info("Audio branch frozen")
    
    def unfreeze_audio_branch(self):
        """Unfreeze audio branch parameters"""
        for param in self.audio_branch.parameters():
            param.requires_grad = True
        logging.info("Audio branch unfrozen")
    
    def enable_contrastive(self):
        """Enable contrastive learning"""
        self.contrastive_enabled = True
        if self.contrastive_head is not None:
            for param in self.contrastive_head.parameters():
                param.requires_grad = True
        logging.info("Contrastive learning enabled")
    
    def disable_contrastive(self):
        """Disable contrastive learning"""
        self.contrastive_enabled = False
        if self.contrastive_head is not None:
            for param in self.contrastive_head.parameters():
                param.requires_grad = False
        logging.info("Contrastive learning disabled")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'audio_embedding_dim': self.model_config['audio_branch']['embedding_dim'],
            'emg_embedding_dim': self.model_config['emg_branch']['embedding_dim'],
            'contrastive_enabled': self.contrastive_enabled
        }
        
        # Add fusion dim only if fusion is enabled
        if self.model_config.get('fusion') is not None:
            model_info['fusion_dim'] = self.model_config['fusion']['projection_dim']
        else:
            model_info['fusion_dim'] = self.model_config['audio_branch']['embedding_dim']
        
        return model_info


def create_model(config: Dict[str, Any]) -> MultiModalDeepfakeDetector:
    """
    Create multi-modal deepfake detection model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model = MultiModalDeepfakeDetector(config)
    
    # Log model info
    model_info = model.get_model_info()
    logging.info(f"Model created with {model_info['total_parameters']:,} total parameters")
    logging.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    return model


if __name__ == "__main__":
    # Test the complete model
    from features import AudioFeatureExtractor, EMGFeatureExtractor
    import yaml
    
    # Load config
    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create components
    audio_extractor = AudioFeatureExtractor(config)
    emg_extractor = EMGFeatureExtractor(config)
    model = create_model(config)
    
    # Test with dummy data
    batch_size = 4
    sample_rate = config['data']['sample_rate']
    duration = config['data']['duration']
    audio_samples = int(sample_rate * duration)
    emg_samples = int(config['data']['emg_sample_rate'] * duration)
    
    audio = torch.randn(batch_size, audio_samples)
    emg = torch.randn(batch_size, 8, emg_samples)
    
    # Extract features
    audio_features_dict = audio_extractor(audio)
    audio_features = audio_extractor.get_feature_tensor(audio_features_dict)
    
    emg_features_dict = emg_extractor(emg)
    emg_features = emg_extractor.get_feature_tensor(emg_features_dict)
    
    # Forward pass
    outputs = model(audio_features, emg_features, return_embeddings=True)
    
    print(f"Audio features shape: {audio_features.shape}")
    print(f"EMG features shape: {emg_features.shape}")
    print(f"Audio embedding shape: {outputs['audio_embedding'].shape}")
    print(f"EMG embedding shape: {outputs['emg_embedding'].shape}")
    print(f"Fused embedding shape: {outputs['fused_embedding'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Test loss computation
    labels = torch.randint(0, 2, (batch_size,)).float()
    losses = model.compute_loss(outputs, labels, contrastive_weight=0.3)
    
    print(f"Classification loss: {losses['classification_loss'].item():.4f}")
    if 'contrastive_loss' in losses:
        print(f"Contrastive loss: {losses['contrastive_loss'].item():.4f}")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    
    print("Multi-modal model created successfully!")
