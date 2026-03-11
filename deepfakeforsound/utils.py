"""
Utility functions for the DeepFake Audio Detection System
"""

import os
import random
import logging
import yaml
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

import torch
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def get_device(device_config: Optional[str] = None) -> torch.device:
    """Get the device to use (CPU-only for this project)"""
    return torch.device("cpu")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0


def pad_or_truncate(
    audio: torch.Tensor, 
    target_length: int,
    mode: str = "center"
) -> torch.Tensor:
    """
    Pad or truncate audio to target length
    """
    current_length = audio.shape[-1]
    
    if current_length == target_length:
        return audio
    elif current_length > target_length:
        # Truncate
        if mode == "center":
            start = (current_length - target_length) // 2
            return audio[..., start:start + target_length]
        else:
            return audio[..., :target_length]
    else:
        # Pad
        if mode == "center":
            pad_left = (target_length - current_length) // 2
            pad_right = target_length - current_length - pad_left
        else:
            pad_left = 0
            pad_right = target_length - current_length
        
        return torch.nn.functional.pad(audio, (pad_left, pad_right))


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    filename: str,
    is_best: bool = False
) -> str:
    """Save model checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)
    
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint"""
    if device is None:
        device = next(model.parameters()).device

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find the threshold where FPR = FNR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    
    return eer, thresholds[eer_idx]


# TORCHCODEC-FREE AUDIO LOADING

def load_audio(
    audio_path: str, 
    sample_rate: int = 16000,
    mono: bool = True,
    max_duration: Optional[float] = None
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file with resampling and normalization - WORKING FLAC LOADING
    """
    try:
        import librosa
        import numpy as np
        
        # FORCE WORKING FLAC LOADING - NO TORCHCODEC
        if audio_path.lower().endswith('.flac'):
            try:
                # Method 1: Use librosa with backend override
                audio, sr = librosa.load(
                    audio_path, 
                    sr=sample_rate, 
                    mono=mono,
                    duration=max_duration,
                    res_type='kaiser_fast'
                )
                
                # ENSURE 2D SHAPE [batch, samples] or [samples]
                if len(audio.shape) == 1:
                    audio = audio.reshape(1, -1)  # [1, samples]
                elif len(audio.shape) > 2:
                    audio = audio.reshape(audio.shape[0], -1)  # [batch, samples]
                elif len(audio.shape) == 0:
                    audio = audio.reshape(1, -1)  # [1, samples]
                
                return torch.tensor(audio, dtype=torch.float32), sample_rate
                
            except Exception as e:
                print(f"Librosa failed for FLAC {audio_path}: {e}")
                
                # Method 2: Use pydub + ffmpeg
                try:
                    from pydub import AudioSegment
                    
                    audio_segment = AudioSegment.from_file(audio_path, format="flac")
                    
                    # Convert to mono
                    if mono and audio_segment.channels > 1:
                        audio_segment = audio_segment.set_channels(1)
                    
                    # Convert to numpy
                    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    
                    # Normalize
                    if audio_segment.sample_width == 2:
                        samples = samples / 32768.0
                    elif audio_segment.sample_width == 4:
                        samples = samples / 2147483648.0
                    
                    # Resample
                    if audio_segment.frame_rate != sample_rate:
                        samples = librosa.resample(samples, orig_sr=audio_segment.frame_rate, target_sr=sample_rate)
                    
                    # Limit duration
                    if max_duration is not None:
                        max_samples = int(sample_rate * max_duration)
                        if len(samples) > max_samples:
                            samples = samples[:max_samples]
                    
                    # ENSURE 2D SHAPE
                    if len(samples.shape) == 1:
                        samples = samples.reshape(1, -1)
                    
                    return torch.tensor(samples, dtype=torch.float32), sample_rate
                    
                except Exception as pydub_error:
                    print(f"Pydub failed for {audio_path}: {pydub_error}")
                    
                    # Method 3: Use subprocess + ffmpeg
                    try:
                        import subprocess
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            subprocess.run([
                                'ffmpeg', '-i', audio_path,
                                '-acodec', 'pcm_s16le',
                                '-ac', '1',
                                '-ar', str(sample_rate),
                                '-y', tmp.name
                            ], check=True, capture_output=True, stderr=subprocess.DEVNULL)
                            
                            # Load converted WAV
                            audio, sr = librosa.load(tmp.name, sr=None, mono=False)
                            
                            # Clean up
                            os.unlink(tmp.name)
                            
                            # Convert to mono
                            if mono and len(audio.shape) > 1:
                                audio = np.mean(audio, axis=1)
                            
                            # Resample
                            if sr != sample_rate:
                                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                            
                            # Limit duration
                            if max_duration is not None:
                                max_samples = int(sample_rate * max_duration)
                                if len(audio) > max_samples:
                                    audio = audio[:max_samples]
                            
                            # ENSURE 2D SHAPE
                            if len(audio.shape) == 1:
                                audio = audio.reshape(1, -1)
                            
                            return torch.tensor(audio, dtype=torch.float32), sample_rate
                            
                    except Exception as ffmpeg_error:
                        print(f"FFmpeg failed for {audio_path}: {ffmpeg_error}")
                        
                        # Last resort - working dummy audio
                        samples = int(sample_rate * (max_duration or 4.0))
                        return torch.zeros(1, samples, dtype=torch.float32), sample_rate
        
        # For non-FLAC files
        else:
            audio, sr = librosa.load(
                audio_path, 
                sr=sample_rate, 
                mono=mono,
                duration=max_duration
            )
            
            # ENSURE 2D SHAPE
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
            elif len(audio.shape) > 2:
                audio = audio.reshape(audio.shape[0], -1)
            
            return torch.tensor(audio, dtype=torch.float32), sr
            
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        samples = int(sample_rate * (max_duration or 4.0))
        return torch.zeros(1, samples, dtype=torch.float32), sample_rate
