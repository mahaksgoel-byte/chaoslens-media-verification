"""
Audio branch of the multi-modal deepfake detection system
2D CNN backbone with transformer encoder and spectral attention
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, efficientnet_b0


class SpectralAttention(nn.Module):
    """
    Spectral attention mechanism for focusing on important frequency regions
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        temperature: float = 0.1
    ):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.temperature = temperature
        
        # Attention layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral attention
        
        Args:
            x: Input tensor [batch_size, channels, freq, time]
            
        Returns:
            Attention-weighted tensor
        """
        # Global average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        # Apply attention
        return x * attention


class ResidualBlock(nn.Module):
    """
    Residual block with spectral attention
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_attention: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention
        
        # Main convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Spectral attention
        if use_attention:
            self.attention = SpectralAttention(out_channels)
        
        # Skip connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention
        if self.use_attention:
            out = self.attention(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class AudioCNNBackbone(nn.Module):
    """
    2D CNN backbone for audio spectrogram processing
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        backbone: str = "resnet18",
        pretrained: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone
        self.use_attention = use_attention
        
        # SIMPLE CNN - replaces complex ResNet for better generalization
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        embedding_dim = 128
        
        self.embedding_dim = embedding_dim
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN backbone
        
        Args:
            x: Input spectrogram [batch_size, channels, freq, time]
            
        Returns:
            Audio embedding [batch_size, embedding_dim]
        """
        batch_size = x.shape[0]
        
        # Handle 3D input [batch, freq, time]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension [batch, 1, freq, time]
        
        # Handle 4D input with different channel count
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)  # Average channels to get [batch, 1, freq, time]
        
        # Forward through simple CNN
        features = self.conv_layers(x)  # [batch, 128, 1, 1]
        features = features.view(batch_size, -1)  # [batch, 128]
        
        return features


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for temporal modeling
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head self-attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear layer
        output = self.fc(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal modeling
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadSelfAttention(d_model, n_heads, dropout),
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model)
                )
            ])
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward through transformer layers
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Transformed tensor [batch_size, seq_len, d_model]
        """
        for attention, ff in self.layers:
            # Self-attention
            x = attention(x, mask)
            
            # Feed-forward
            ff_output = ff(x)
            x = x + ff_output
        
        return x


class AudioBranch(nn.Module):
    """
    Complete audio branch with CNN backbone and transformer encoder
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        input_channels: int = 1
    ):
        super().__init__()
        self.config = config['model']['audio_branch']
        
        # CNN backbone
        self.cnn_backbone = AudioCNNBackbone(
            input_channels=input_channels,
            backbone=self.config['backbone'],
            pretrained=self.config['pretrained'],
            use_attention=True
        )
        
        # Projection to transformer dimension
        self.cnn_embedding_dim = self.cnn_backbone.embedding_dim
        self.transformer_d_model = self.config['transformer']['d_model']
        
        self.cnn_projection = nn.Sequential(
            nn.Linear(self.cnn_embedding_dim, self.transformer_d_model),
            nn.ReLU(),
            nn.Dropout(self.config['transformer']['dropout'])
        )
        
        # Positional encoding for temporal modeling
        self.positional_encoding = PositionalEncoding(
            self.transformer_d_model,
            dropout=self.config['transformer']['dropout']
        )
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=self.transformer_d_model,
            n_heads=self.config['transformer']['n_heads'],
            n_layers=self.config['transformer']['n_layers'],
            dropout=self.config['transformer']['dropout']
        )
        
        # Final embedding projection
        self.embedding_dim = self.config['embedding_dim']
        self.final_projection = nn.Sequential(
            nn.Linear(self.transformer_d_model, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.config['transformer']['dropout'])
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through audio branch
        
        Args:
            audio_features: Audio features [batch_size, channels, freq, time]
            
        Returns:
            Audio embedding [batch_size, embedding_dim]
        """
        # CNN backbone forward
        embedding = self.cnn_backbone(audio_features)
        
        # Project to transformer dimension
        embedding = self.cnn_projection(embedding)
        
        # Add batch dimension for transformer (sequence length = 1)
        embedding = embedding.unsqueeze(1)  # [batch, 1, d_model]
        
        # Positional encoding
        embedding = self.positional_encoding(embedding)
        
        # Transformer encoder
        embedding = self.transformer_encoder(embedding)
        
        # Remove sequence dimension and project to final embedding
        embedding = embedding.squeeze(1)  # [batch, d_model]
        embedding = self.final_projection(embedding)  # [batch, embedding_dim]
        
        return embedding


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def create_audio_branch(config: Dict[str, Any], input_channels: int = 1) -> AudioBranch:
    """
    Create audio branch from configuration
    
    Args:
        config: Configuration dictionary
        input_channels: Number of input channels
        
    Returns:
        Audio branch instance
    """
    return AudioBranch(config, input_channels)


if __name__ == "__main__":
    # Test the audio branch
    from features import AudioFeatureExtractor
    import yaml
    
    # Load config
    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create components
    audio_extractor = AudioFeatureExtractor(config)
    audio_branch = create_audio_branch(config)
    
    # Test with dummy data
    batch_size = 4
    sample_rate = config['data']['sample_rate']
    duration = config['data']['duration']
    samples = int(sample_rate * duration)
    
    audio = torch.randn(batch_size, samples)
    
    # Extract features
    features = audio_extractor(audio)
    audio_tensor = audio_extractor.get_feature_tensor(features)
    
    # Forward through audio branch
    embedding = audio_branch(audio_tensor)
    
    print(f"Audio tensor shape: {audio_tensor.shape}")
    print(f"Audio embedding shape: {embedding.shape}")
    print(f"Audio branch created successfully!")
