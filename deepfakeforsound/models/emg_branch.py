"""
EMG branch of the multi-modal deepfake detection system
1D ResNet encoder with temporal transformer for muscle activation patterns
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """
    1D convolutional block with batch norm and activation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))


class ResidualBlock1D(nn.Module):
    """
    1D residual block for EMG signal processing
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Main path
        self.conv1 = ConvBlock1D(in_channels, out_channels, 3, stride, 1, dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for EMG signal processing
    """
    
    def __init__(
        self,
        input_channels: int = 8,  # Typical EMG channels
        layers: list = [2, 2, 2, 2],
        channels: list = [64, 128, 256, 512],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.in_channels = channels[0]
        
        # Initial convolution
        self.conv1 = ConvBlock1D(input_channels, channels[0], 7, 2, 3, dropout)
        self.maxpool = nn.MaxPool1d(3, 2, 1)
        
        # Residual layers
        self.layer1 = self._make_layer(channels[0], layers[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(channels[1], layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(channels[3], layers[3], stride=2, dropout=dropout)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.embedding_dim = channels[-1]
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int, dropout: float) -> nn.Sequential:
        """Create a residual layer"""
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, dropout))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, dropout=dropout))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 1D ResNet
        
        Args:
            x: EMG tensor [batch_size, channels, time]
            
        Returns:
            EMG embedding [batch_size, embedding_dim]
        """
        # Initial layers
        x = self.conv1(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention for EMG sequences
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 6,
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
        Apply temporal attention
        
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


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal EMG modeling
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 6,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                TemporalAttention(d_model, n_heads, dropout),
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
        Forward through temporal transformer layers
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Transformed tensor [batch_size, seq_len, d_model]
        """
        for attention, ff in self.layers:
            # Temporal attention
            x = attention(x, mask)
            
            # Feed-forward
            ff_output = ff(x)
            x = x + ff_output
        
        return x


class PositionalEncoding1D(nn.Module):
    """
    Positional encoding for 1D sequences
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


class EMGBranch(nn.Module):
    """
    Complete EMG branch with 1D ResNet and temporal transformer
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        input_channels: int = 8
    ):
        super().__init__()
        self.config = config['model']['emg_branch']
        
        # Check if EMG is disabled
        if self.config.get('backbone') == 'none' or self.config.get('transformer') is None:
            # Create dummy EMG branch that returns zeros
            self.embedding_dim = 1
            self.disabled = True
            return
        
        self.disabled = False
        
        # 1D ResNet backbone
        self.resnet1d = ResNet1D(
            input_channels=input_channels,
            layers=[2, 2, 2, 2],
            channels=[64, 128, 256, 512],
            dropout=0.2
        )
        
        # Projection to transformer dimension
        self.resnet_embedding_dim = self.resnet1d.embedding_dim
        self.transformer_d_model = self.config['transformer']['d_model']
        
        self.resnet_projection = nn.Sequential(
            nn.Linear(self.resnet_embedding_dim, self.transformer_d_model),
            nn.ReLU(),
            nn.Dropout(self.config['transformer']['dropout'])
        )
        
        # Positional encoding for temporal modeling
        self.positional_encoding = PositionalEncoding1D(
            self.transformer_d_model,
            dropout=self.config['transformer']['dropout']
        )
        
        # Temporal transformer encoder
        self.temporal_transformer = TemporalTransformerEncoder(
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
    
    def forward(self, emg_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EMG branch
        
        Args:
            emg_features: EMG features [batch_size, channels, time]
            
        Returns:
            EMG embedding [batch_size, embedding_dim]
        """
        # Return dummy embedding if disabled
        if self.disabled:
            batch_size = emg_features.shape[0] if emg_features is not None else 1
            return torch.zeros(batch_size, self.embedding_dim, device=emg_features.device if emg_features is not None else 'cpu')
        
        # Original forward logic
        batch_size = emg_features.shape[0]
        
        # ResNet backbone forward
        embedding = self.resnet1d(emg_features)
        
        # Project to transformer dimension
        embedding = self.resnet_projection(embedding)
        
        # Add sequence dimension for transformer
        embedding = embedding.unsqueeze(1)  # [batch, 1, d_model]
        
        # Positional encoding
        embedding = self.positional_encoding(embedding)
        
        # Temporal transformer
        embedding = self.temporal_transformer(embedding)
        
        # Remove sequence dimension and project to final embedding
        embedding = embedding.squeeze(1)  # [batch, d_model]
        embedding = self.final_projection(embedding)  # [batch, embedding_dim]
        
        return embedding
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, emg_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EMG branch
        
        Args:
            emg_features: EMG features [batch_size, channels, time]
            
        Returns:
            EMG embedding [batch_size, embedding_dim]
        """
        batch_size, channels, time = emg_features.shape
        
        # Process EMG through 1D ResNet
        # We'll process overlapping windows to capture temporal patterns
        window_size = min(512, time)  # Use 512-sample windows or smaller if needed
        stride = window_size // 4  # 75% overlap
        
        if time <= window_size:
            # Single window processing
            resnet_features = self.resnet1d(emg_features)  # [batch, resnet_embedding_dim]
            resnet_features = resnet_features.unsqueeze(1)  # [batch, 1, resnet_embedding_dim]
        else:
            # Multi-window processing
            windows = []
            for i in range(0, time - window_size + 1, stride):
                window = emg_features[:, :, i:i + window_size]
                window_features = self.resnet1d(window)  # [batch, resnet_embedding_dim]
                windows.append(window_features)
            
            resnet_features = torch.stack(windows, dim=1)  # [batch, n_windows, resnet_embedding_dim]
        
        # Project to transformer dimension
        resnet_features = self.resnet_projection(resnet_features)  # [batch, seq_len, transformer_d_model]
        
        # Add positional encoding
        resnet_features = self.positional_encoding(resnet_features)
        
        # Temporal transformer encoding
        transformer_features = self.temporal_transformer(resnet_features)  # [batch, seq_len, transformer_d_model]
        
        # Global temporal pooling (mean and max)
        mean_pooled = torch.mean(transformer_features, dim=1)  # [batch, transformer_d_model]
        max_pooled = torch.max(transformer_features, dim=1)[0]  # [batch, transformer_d_model]
        
        # Combine pooling results
        pooled_features = mean_pooled + max_pooled
        
        # Final projection
        emg_embedding = self.final_projection(pooled_features)  # [batch, embedding_dim]
        
        return emg_embedding


def create_emg_branch(config: Dict[str, Any], input_channels: int = 8) -> EMGBranch:
    """
    Create EMG branch from configuration
    
    Args:
        config: Configuration dictionary
        input_channels: Number of EMG channels
        
    Returns:
        EMG branch instance
    """
    return EMGBranch(config, input_channels)


if __name__ == "__main__":
    # Test the EMG branch
    from features import EMGFeatureExtractor
    import yaml
    
    # Load config
    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create components
    emg_extractor = EMGFeatureExtractor(config)
    emg_branch = create_emg_branch(config)
    
    # Test with dummy data
    batch_size = 4
    emg_sample_rate = config['data']['emg_sample_rate']
    duration = config['data']['duration']
    samples = int(emg_sample_rate * duration)
    channels = 8  # Typical EMG channels
    
    emg = torch.randn(batch_size, channels, samples)
    
    # Extract features
    features = emg_extractor(emg)
    emg_tensor = emg_extractor.get_feature_tensor(features)
    
    # Forward through EMG branch
    embedding = emg_branch(emg_tensor)
    
    print(f"EMG tensor shape: {emg_tensor.shape}")
    print(f"EMG embedding shape: {embedding.shape}")
    print(f"EMG branch created successfully!")
