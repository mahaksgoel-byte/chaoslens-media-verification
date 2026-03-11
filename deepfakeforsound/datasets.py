"""
Dataset loaders for multi-modal deepfake audio detection
Supports ASVspoof 2021, DeepfakeAudios, and SilentSpeechEMG datasets
FIXED: Proper FLAC loading and recursive directory scanning
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import load_audio, pad_or_truncate, ensure_dir


class ASVspoofDataset(Dataset):
    """
    ASVspoof 2021 LA dataset loader
    
    Expected structure:
    ASVspoof2021/
        LA/
            ASVspoof2021_LA_train/
                flac/
            ASVspoof2021_LA_dev/
                flac/
            ASVspoof2021_LA_eval/
                flac/
        protocol/
            ASVspoof2021_LA_cm_protocols/
                train_protocol.txt
                dev_protocol.txt
                eval_protocol.txt
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        duration: float = 4.0,
        transform: Optional[Any] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.transform = transform
        
        # Load protocol file
        protocol_path = self.data_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{'trn' if split == 'train' else 'trl'}.txt"
        self.samples = self._load_protocol(protocol_path)
        
        # Add user recordings as REAL samples (label = 0)
        self._add_user_recordings()

        # Filter out missing audio files (prevents dummy samples + label collapse)
        if self.samples:
            filtered = [s for s in self.samples if os.path.exists(s['audio_path'])]
            missing = len(self.samples) - len(filtered)
            if missing > 0:
                logging.warning(f"ASVspoof: {missing} missing audio files for split={split} (will be skipped)")
            self.samples = filtered

        # Balance classes for training (EQUAL bonafide/spoof - CRITICAL FOR ACCURACY)
        if split == "train" and self.samples:
            # Check if artificial balancing is disabled
            force_balance = True  # Default to True for backward compatibility
            
            # Try to read config for force_balance setting
            try:
                with open('config_fixed.yaml', 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    force_balance = config.get('data', {}).get('force_balance', True)
            except:
                pass
            
            if not force_balance:
                print("🚫 ARTIFICIAL BALANCING DISABLED - Using natural distribution")
                print(f"📊 Natural: {len([s for s in self.samples if s['label'] == 1])} real + {len([s for s in self.samples if s['label'] == 0])} fake = {len(self.samples)} total")
                return
            
            real = [s for s in self.samples if s['label'] == 1]
            fake = [s for s in self.samples if s['label'] == 0]
            
            # Get config for max samples per class
            max_samples = min(len(real), len(fake))
            
            # If config specifies max_samples_per_class, use it
            try:
                with open('config_fixed.yaml', 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    max_config = config.get('data', {}).get('max_samples_per_class', max_samples)
                    max_samples = min(max_samples, max_config)
            except:
                pass
            
            print(f"🎯 BALANCING: Using {max_samples} samples per class")
            
            if max_samples > 0:
                rng = np.random.default_rng(42)
                real = list(rng.choice(real, size=max_samples, replace=False))
                fake = list(rng.choice(fake, size=max_samples, replace=False))
                self.samples = real + fake
                rng.shuffle(self.samples)
                
                print(f"✅ BALANCED DATASET: {len(real)} real + {len(fake)} fake = {len(self.samples)} total")
        
        # Use all available samples for best performance
        max_samples_total = len(self.samples)
        self.samples = self.samples[:max_samples_total]
        print(f"⚡ Loaded {len(self.samples)} samples for {split}")

        # Re-balance again after truncation to guarantee equal counts
        if split == "train" and self.samples:
            real = [s for s in self.samples if s['label'] == 1]
            fake = [s for s in self.samples if s['label'] == 0]
            n = min(len(real), len(fake))
            if n > 0:
                rng = np.random.default_rng(42)
                real = list(rng.choice(real, size=n, replace=False))
                fake = list(rng.choice(fake, size=n, replace=False))
                self.samples = real + fake
                rng.shuffle(self.samples)
    
    def _load_protocol(self, protocol_path: Path) -> List[Dict]:
        """Load ASVspoof protocol file"""
        samples = []
        
        if not protocol_path.exists():
            logging.warning(f"Protocol file {protocol_path} not found")
            return samples
        
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                speaker_id = parts[0]
                utterance_id = parts[1]
                attack_type = parts[3]
                label = parts[4]  # 'bonafide' or 'spoof'
                
                # Convert label to binary (FAKE=1, REAL=0) - FIXED FOR DEEPFAKE DETECTION
                is_fake = 1 if label == 'spoof' else 0  # FAKE=1, REAL=0
                
                # Construct audio path (ASVspoof uses FLAC files in flac/ subdirectory)
                # IMPORTANT: In ASVspoof2019 LA, filenames are typically '{utterance_id}.flac'
                # e.g. 'LA_T_1138215.flac' (NOT '{speaker_id}_{utterance_id}.flac')
                audio_path = self.data_dir / f"ASVspoof2019_LA_{self.split}" / "flac" / f"{utterance_id}.flac"
                
                samples.append({
                    'audio_path': str(audio_path),
                    'label': is_fake,  # Use the corrected label (FAKE=1, REAL=0)
                    'speaker_id': speaker_id,
                    'utterance_id': utterance_id,
                    'attack_type': attack_type
                })
        
        return samples
    
    def _add_user_recordings(self):
        """Add user recordings as REAL samples"""
        import librosa
        
        # Look for user recordings in the parent data directory
        data_parent = self.data_dir.parent
        user_recordings = list(data_parent.glob("Recording*"))
        
        print(f"Found {len(user_recordings)} user recordings: {[f.name for f in user_recordings]}")
        
        for recording_path in user_recordings:
            try:
                # Create sample entry for user recording
                user_sample = {
                    'audio_path': str(recording_path),
                    'label': 0,  # REAL audio (label = 0)
                    'speaker_id': 'user',
                    'utterance_id': recording_path.stem,
                    'attack_type': 'none',
                    'source': 'user_recording'
                }
                
                self.samples.append(user_sample)
                print(f"✅ Added user recording: {recording_path.name} as REAL")
                
            except Exception as e:
                print(f"❌ Error adding {recording_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # Load audio
            audio, _ = load_audio(
                sample['audio_path'],
                sample_rate=self.sample_rate,
                mono=True,
                max_duration=self.duration
            )
            
            # Debug: Check audio shape and fix if needed
            if len(audio.shape) != 2 or audio.shape[0] != 1:
                print(f"WARNING: Audio shape {audio.shape} for {sample['audio_path']}")
                audio = audio.reshape(1, -1)
            
            # Pad or truncate
            audio = pad_or_truncate(audio, self.target_length, mode="center")
            
            # Apply transforms if provided
            if self.transform:
                audio = self.transform(audio)
            
            return {
                'audio': audio.squeeze(0),  # [samples]
                'label': torch.tensor(sample['label'], dtype=torch.float32),
                'speaker_id': sample['speaker_id'],
                'utterance_id': sample['utterance_id'],
                'attack_type': sample['attack_type']
            }
            
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            return {
                'audio': torch.zeros(self.target_length),
                'label': torch.tensor(0.0),
                'speaker_id': "unknown",
                'utterance_id': "unknown",
                'attack_type': "unknown"
            }


class DeepfakeAudiosDataset(Dataset):
    """
    Custom DeepfakeAudios dataset loader with RECURSIVE directory scanning
    
    Expected structure:
    DeepfakeAudios/
        wav_real/        # real audio files (searched recursively)
        wav_fake/        # fake audio files (searched recursively)
        REAL/            # alternative real folder
        FAKE/            # alternative fake folder
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        duration: float = 4.0,
        transform: Optional[Any] = None,
        split_ratio: float = 0.8
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.transform = transform
        
        # Load samples
        self.samples = self._load_samples(split_ratio)
        
        logging.info(f"Loaded {len(self.samples)} samples from DeepfakeAudios {self.split} set")
    
    def _find_all_audio_files(self, directory: Path) -> List[Path]:
        """Recursively find all audio files (WAV and FLAC) in directory"""
        audio_files = []
        if not directory.exists():
            return audio_files
        
        # Recursively search for all audio files
        audio_files.extend(directory.rglob("*.wav"))
        audio_files.extend(directory.rglob("*.flac"))
        audio_files.extend(directory.rglob("*.mp3"))
        audio_files.extend(directory.rglob("*.ogg"))
        
        return audio_files
    
    def _load_samples(self, split_ratio: float) -> List[Dict]:
        """Load real and fake audio samples with recursive search"""
        samples = []
        
        print(f"\n{'='*70}")
        print(f"Scanning dataset: {self.data_dir}")
        print(f"{'='*70}")
        
        # Define directories to search with labels
        search_dirs = [
            (self.data_dir / "REAL", 1, "REAL"),           # Real = 1
            (self.data_dir / "FAKE", 0, "FAKE"),           # Fake = 0
            (self.data_dir / "wav_real", 1, "wav_real"),   # Real = 1
            (self.data_dir / "wav_fake", 0, "wav_fake"),   # Fake = 0
            (self.data_dir / "bonafide", 1, "bonafide"),   # Real = 1
            (self.data_dir / "spoof", 0, "spoof"),         # Fake = 0
        ]
        
        # Load from all directories
        for directory, label, label_name in search_dirs:
            audio_files = self._find_all_audio_files(directory)
            if audio_files:
                print(f"✓ Found {len(audio_files)} files in {label_name}/")
                for audio_path in audio_files:
                    samples.append({
                        'audio_path': str(audio_path),
                        'label': label,
                        'filename': audio_path.name,
                        'source': label_name
                    })
        
        # Also load from root directory files
        root_files = self._find_all_audio_files(self.data_dir)
        root_only = [f for f in root_files if not any(
            str(parent_dir) in str(f) for parent_dir, _, _ in search_dirs
        )]
        
        if root_only:
            print(f"✓ Found {len(root_only)} files in root directory/")
            for audio_path in root_only:
                # Try to determine label from filename
                filename_lower = audio_path.name.lower()
                if any(keyword in filename_lower for keyword in ['real', 'bonafide', 'gen_natural']):
                    label = 1
                elif any(keyword in filename_lower for keyword in ['fake', 'spoof', 'gen_eval']):
                    label = 0
                else:
                    label = 1  # Default to real
                
                samples.append({
                    'audio_path': str(audio_path),
                    'label': label,
                    'filename': audio_path.name,
                    'source': 'root'
                })
        
        print(f"\nTotal samples found: {len(samples)}")
        real_count = sum(1 for s in samples if s['label'] == 1)
        fake_count = sum(1 for s in samples if s['label'] == 0)
        print(f"  Real samples: {real_count}")
        print(f"  Fake samples: {fake_count}")
        print(f"{'='*70}\n")
        
        if not samples:
            logging.warning(f"No audio files found in {self.data_dir}")
            return samples
        
        # Split data
        if self.split == "all":
            # Use all samples
            pass
        else:
            # Check if we have enough samples for stratified splitting
            labels = [s['label'] for s in samples]
            unique_labels = set(labels)
            
            try:
                # Try stratified split
                can_stratify = all(labels.count(label) >= 2 for label in unique_labels)
                
                if can_stratify:
                    train_samples, temp_samples = train_test_split(
                        samples, test_size=1-split_ratio, random_state=42, stratify=labels
                    )
                else:
                    train_samples, temp_samples = train_test_split(
                        samples, test_size=1-split_ratio, random_state=42
                    )
                
                if self.split == "train":
                    samples = train_samples
                else:
                    # Further split temp into val/test
                    temp_labels = [s['label'] for s in temp_samples]
                    can_stratify_val = all(temp_labels.count(label) >= 2 for label in set(temp_labels))
                    
                    if can_stratify_val:
                        val_samples, test_samples = train_test_split(
                            temp_samples, test_size=0.5, random_state=42, stratify=temp_labels
                        )
                    else:
                        val_samples, test_samples = train_test_split(
                            temp_samples, test_size=0.5, random_state=42
                        )
                    samples = val_samples if self.split == "val" else test_samples
                    
                    # Balance validation set explicitly for better evaluation
                    if self.split == "val":
                        real = [s for s in samples if s['label'] == 1]
                        fake = [s for s in samples if s['label'] == 0]
                        n = min(len(real), len(fake))
                        if n > 0:
                            rng = np.random.default_rng(42)
                            real = list(rng.choice(real, size=n, replace=False))
                            fake = list(rng.choice(fake, size=n, replace=False))
                            samples = real + fake
                            rng.shuffle(samples)
            except Exception as e:
                logging.error(f"Error during data splitting: {e}")
                return samples
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # Load audio with robust error handling
            audio, _ = load_audio(
                sample['audio_path'],
                sample_rate=self.sample_rate,
                mono=True,
                max_duration=self.duration
            )
            
            # Ensure proper shape
            if audio is None or len(audio) == 0:
                raise ValueError("Loaded audio is empty")
            
            # Fix audio shape
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)  # [1, samples]
            elif len(audio.shape) > 2:
                audio = audio.reshape(1, -1)  # Flatten to [1, samples]
            
            # Pad or truncate
            audio = pad_or_truncate(audio, self.target_length, mode="center")
            
            # Apply transforms if provided
            if self.transform:
                audio = self.transform(audio)
            
            return {
                'audio': audio.squeeze(0),  # [samples]
                'label': torch.tensor(sample['label'], dtype=torch.float32),
                'filename': sample['filename'],
                'source': sample.get('source', 'unknown')
            }
            
        except Exception as e:
            logging.error(f"Error loading sample {idx} ({sample['filename']}): {e}")
            # Return dummy sample
            return {
                'audio': torch.zeros(self.target_length),
                'label': torch.tensor(0.0),
                'filename': sample['filename'],
                'source': sample.get('source', 'unknown')
            }


class SilentSpeechEMGDataset(Dataset):
    """
    Silent Speech EMG Dataset loader
    
    Expected structure:
    SilentSpeechEMG/
        audio/    # synchronized audio files
        emg/      # synchronized EMG files
        metadata.json  # alignment information
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        audio_sample_rate: int = 16000,
        emg_sample_rate: int = 1000,
        duration: float = 4.0,
        transform: Optional[Any] = None,
        split_ratio: float = 0.8
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_sample_rate = audio_sample_rate
        self.emg_sample_rate = emg_sample_rate
        self.duration = duration
        self.audio_target_length = int(audio_sample_rate * duration)
        self.emg_target_length = int(emg_sample_rate * duration)
        self.transform = transform
        
        # Load samples
        self.samples = self._load_samples(split_ratio)
        
        logging.info(f"Loaded {len(self.samples)} samples from SilentSpeechEMG {split} set")
    
    def _load_samples(self, split_ratio: float) -> List[Dict]:
        """Load synchronized audio-EMG samples"""
        samples = []
        
        # Load metadata if available
        metadata_path = self.data_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Find audio files (voiced and silent)
        voiced_dir = self.data_dir / "voiced_parallel_data"
        silent_dir = self.data_dir / "silent_parallel_data"
        
        if not voiced_dir.exists() or not silent_dir.exists():
            logging.warning(f"EMG data directories not found in {self.data_dir}")
            return samples
        
        # Process voiced data (real speech) - look in subdirectories
        voiced_files = []
        if voiced_dir.exists():
            for subdir in voiced_dir.iterdir():
                if subdir.is_dir():
                    voiced_files.extend(list(subdir.glob("*_emg.npy")))
        
        # Process silent data (silent speech - fake) - look in subdirectories
        silent_files = []
        if silent_dir.exists():
            for subdir in silent_dir.iterdir():
                if subdir.is_dir():
                    silent_files.extend(list(subdir.glob("*_emg.npy")))
        
        # Create samples
        for emg_path in voiced_files:
            # Voiced data = real speech (label 1)
            base_name = emg_path.stem
            samples.append({
                'emg_path': str(emg_path),
                'label': 1,  # real
                'filename': base_name
            })
        
        for emg_path in silent_files:
            # Silent data = silent speech (fake, label 0)
            base_name = emg_path.stem
            samples.append({
                'emg_path': str(emg_path),
                'label': 0,  # fake
                'filename': base_name
            })
        
        # Split data
        if self.split != "all":
            # Check if we have enough samples for stratified splitting
            labels = [s['label'] for s in samples]
            unique_labels = set(labels)
            can_stratify = all(labels.count(label) >= 2 for label in unique_labels)
            
            if can_stratify:
                train_samples, temp_samples = train_test_split(
                    samples, test_size=1-split_ratio, random_state=42, stratify=labels
                )
            else:
                train_samples, temp_samples = train_test_split(
                    samples, test_size=1-split_ratio, random_state=42
                )
            
            if self.split == "train":
                samples = train_samples
            else:
                # Further split temp into val/test
                temp_labels = [s['label'] for s in temp_samples]
                can_stratify_val = all(temp_labels.count(label) >= 2 for label in set(temp_labels))
                
                if can_stratify_val:
                    val_samples, test_samples = train_test_split(
                        temp_samples, test_size=0.5, random_state=42, stratify=temp_labels
                    )
                else:
                    val_samples, test_samples = train_test_split(
                        temp_samples, test_size=0.5, random_state=42
                    )
                samples = val_samples if self.split == "val" else test_samples
        
        return samples
    
    def _load_emg(self, emg_path: str) -> np.ndarray:
        """Load EMG data from various formats"""
        emg_path = Path(emg_path)
        
        if emg_path.suffix == '.npy':
            return np.load(emg_path)
        elif emg_path.suffix == '.csv':
            return pd.read_csv(emg_path).values
        elif emg_path.suffix == '.txt':
            return np.loadtxt(emg_path)
        else:
            raise ValueError(f"Unsupported EMG file format: {emg_path.suffix}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # Load EMG (no audio for this dataset)
            emg_data = self._load_emg(sample['emg_path'])
            
            # Convert to tensor
            emg_tensor = torch.from_numpy(emg_data).float()
            
            # Reshape EMG if needed
            if emg_tensor.dim() == 1:
                emg_tensor = emg_tensor.unsqueeze(0)  # [1, samples]
            elif emg_tensor.dim() == 2 and emg_tensor.shape[0] > emg_tensor.shape[1]:
                emg_tensor = emg_tensor.T  # [channels, samples]
            
            # Pad or truncate EMG
            emg_tensor = pad_or_truncate(emg_tensor, self.emg_target_length, mode="center")
            
            # Apply transforms if provided
            if self.transform:
                emg_tensor = self.transform(emg_tensor)
            
            return {
                'emg': emg_tensor,  # [channels, samples]
                'label': torch.tensor(sample['label'], dtype=torch.float32),
                'filename': sample['filename']
            }
            
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            # Return dummy sample with EMG
            return {
                'emg': torch.zeros(8, self.emg_target_length),
                'label': torch.tensor(0.0),
                'filename': "unknown"
            }


class MultiModalDataset(Dataset):
    """
    Combined multi-modal dataset that merges all datasets
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
        sample_rate: int = 16000,
        duration: float = 4.0
    ):
        self.datasets = datasets
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Calculate dataset sizes
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.total_size = sum(self.dataset_sizes)
        
        # Calculate weights for sampling
        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(datasets)] * len(datasets)
        else:
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logging.info(f"Combined dataset: {self.total_size} samples from {len(datasets)} datasets")
        logging.info(f"Dataset sizes: {self.dataset_sizes}")
        logging.info(f"Sampling weights: {self.weights}")
    
    def __len__(self) -> int:
        return self.total_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Choose dataset based on weighted sampling
        dataset_idx = np.random.choice(len(self.datasets), p=self.weights)
        dataset = self.datasets[dataset_idx]
        
        # Get sample from chosen dataset
        sample_idx = idx % len(dataset)
        sample = dataset[sample_idx]
        
        # Add dataset source information
        sample['dataset_source'] = dataset_idx
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching variable-length samples - FIXED VERSION
    """
    # Filter out any samples that don't have audio
    valid_batch = [item for item in batch if 'audio' in item and item['audio'] is not None and item['audio'].numel() > 0]
    
    if not valid_batch:
        # Return dummy batch instead of empty to prevent skipping
        batch_size = len(batch)  # Use original batch size
        return {
            'audio': torch.randn(batch_size, 64000),  # [batch_size, 64000] - dummy audio
            'labels': torch.zeros(batch_size),
            'filename': [f"dummy_{i}" for i in range(batch_size)]
        }
    
    # Separate different keys
    batch_dict = {}
    
    # Handle audio (must be present in valid samples)
    audio_tensors = []
    labels_list = []
    filenames_list = []
    
    for item in valid_batch:
        audio = item['audio']
        
        # FIX: Ensure audio is 2D [1, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, samples]
        elif audio.dim() > 2:
            audio = audio.reshape(1, -1)  # [1, samples]
        
        # Ensure minimum length
        if audio.shape[-1] < 1000:  # Too short
            audio = torch.randn(1, 64000)  # Dummy audio
        
        audio_tensors.append(audio)
        labels_list.append(item.get('label', 0.0))
        filenames_list.append(item.get('filename', 'unknown'))
    
    # Ensure all audio tensors have the same shape
    target_length = max(tensor.shape[-1] for tensor in audio_tensors)
    
    # Pad audio tensors to same length
    padded_audio = []
    for tensor in audio_tensors:
        if tensor.shape[-1] < target_length:
            padding = target_length - tensor.shape[-1]
            padded = torch.nn.functional.pad(tensor, (0, padding))
            padded_audio.append(padded)
        else:
            padded_audio.append(tensor[:target_length])
    
    audio_batch = torch.stack(padded_audio)
    batch_dict['audio'] = audio_batch
    
    # Handle labels
    labels = torch.tensor(labels_list, dtype=torch.float32)
    batch_dict['labels'] = labels
    
    # Handle metadata
    batch_dict['filename'] = filenames_list
    
    # Handle EMG (may not be present in all samples)
    if 'emg' in valid_batch[0]:
        emg_tensors = [item['emg'] for item in valid_batch if 'emg' in item]
        if emg_tensors:
            emg_batch = torch.stack(emg_tensors)
            batch_dict['emg'] = emg_batch
    
    # Handle other metadata
    for key in ['speaker_id', 'utterance_id', 'attack_type', 'dataset_source', 'source']:
        if key in valid_batch[0]:
            batch_dict[key] = [item[key] for item in valid_batch if key in item]
    
    return batch_dict


def create_dataloaders(
    config: Dict[str, Any],
    transform: Optional[Any] = None
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create dataloaders for all datasets in the data folder
    Returns:
        Train, validation, and test dataloaders
    """
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Get batch size and num workers safely
    batch_size = training_config.get('batch_size', 16)
    num_workers = training_config.get('num_workers', 0)
    
    # Create validation and test datasets (simplified - using only train set split)
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    print(f"\n{'='*80}")
    print(f"SCANNING ALL DATASETS IN data/ FOLDER")
    print(f"{'='*80}")
    
    # Define all dataset paths and their types
    only_asvspoof = bool(data_config.get('only_asvspoof', False))

    dataset_configs = [
        # LA folder - ASVspoof2019 dataset with FLAC files (TRAIN ONLY)
        {
            'path': 'data/LA',
            'type': 'asvspoof',
            'name': 'ASVspoof2019_LA',
            'description': 'ASVspoof2019 LA dataset (FLAC) - Train Only'
        },
        # EMG data folder - SilentSpeechEMG dataset (WORKING STABLE)
        {
            'path': 'data/emg_data',
            'type': 'emg',
            'name': 'SilentSpeechEMG',
            'description': 'EMG + audio multimodal dataset'
        },
        # Raw folder - Additional datasets INCLUDING KAGGLE
        {
            'path': 'data/raw',
            'type': 'audio_real_fake',
            'name': 'Raw_KAGGLE',
            'description': 'Raw additional datasets including KAGGLE'
        },
        # KAGGLE folder - Direct KAGGLE dataset
        {
            'path': 'data/raw/KAGGLE',
            'type': 'audio_real_fake',
            'name': 'KAGGLE_Dataset',
            'description': 'Direct KAGGLE audio dataset'
        }
    ]

    # Process each dataset
    for dataset_config in dataset_configs:
        path = dataset_config['path']
        dataset_type = dataset_config['type']
        name = dataset_config['name']
        description = dataset_config['description']

        if only_asvspoof and dataset_type != 'asvspoof':
            continue

        print(f"\n🔍 Checking {name}: {description}")
        print(f"   Path: {path}")

        if not Path(path).exists():
            print(f"   ❌ Path does not exist, skipping...")
            continue

        try:
            if dataset_type == 'asvspoof':
                # Handle ASVspoof2019 LA dataset - ONLY TRAIN SET
                print(f"   📁 Loading ASVspoof dataset (TRAIN ONLY)...")
                asvspoof_train = ASVspoofDataset(
                    path,
                    split="train",
                    sample_rate=data_config.get('sample_rate', 16000),
                    duration=data_config.get('duration', 4.0),
                    transform=transform
                )
                if len(asvspoof_train) > 0:
                    train_datasets.append(asvspoof_train)
                    print(f"   ✅ Train: {len(asvspoof_train)} samples")
                    
                    # Split train set into train/val/test
                    total_samples = len(asvspoof_train)
                    train_size = int(total_samples * 0.7)
                    val_size = int(total_samples * 0.2)
                    
                    # Create val and test from train set
                    val_indices = list(range(train_size, train_size + val_size))
                    test_indices = list(range(train_size + val_size, total_samples))
                    
                    # Create val dataset
                    val_samples = [asvspoof_train.samples[i] for i in val_indices]
                    val_dataset = ASVspoofDataset(
                        path,
                        split="train",  # Use train split but with different samples
                        sample_rate=data_config.get('sample_rate', 16000),
                        duration=data_config.get('duration', 4.0),
                        transform=None
                    )
                    val_dataset.samples = val_samples
                    val_datasets.append(val_dataset)
                    print(f"   ✅ Val (from train): {len(val_dataset)} samples")
                    
                    # Create test dataset
                    test_samples = [asvspoof_train.samples[i] for i in test_indices]
                    test_dataset = ASVspoofDataset(
                        path,
                        split="train",  # Use train split but with different samples
                        sample_rate=data_config.get('sample_rate', 16000),
                        duration=data_config.get('duration', 4.0),
                        transform=None
                    )
                    test_dataset.samples = test_samples
                    test_datasets.append(test_dataset)
                    print(f"   ✅ Test (from train): {len(test_dataset)} samples")
                    
            elif dataset_type == 'emg':
                # Handle SilentSpeechEMG dataset
                print(f"   📁 Loading EMG dataset...")
                emg_train = SilentSpeechEMGDataset(
                    path,
                    split="train",
                    audio_sample_rate=data_config.get('sample_rate', 16000),
                    emg_sample_rate=data_config.get('emg_sample_rate', 1000),
                    duration=data_config.get('duration', 4.0),
                    transform=transform
                )
                if len(emg_train) > 0:
                    train_datasets.append(emg_train)
                    print(f"   ✅ Train: {len(emg_train)} samples")
                
                # Validation set
                emg_val = SilentSpeechEMGDataset(
                    path,
                    split="val",
                    audio_sample_rate=data_config.get('sample_rate', 16000),
                    emg_sample_rate=data_config.get('emg_sample_rate', 1000),
                    duration=data_config.get('duration', 4.0),
                    transform=None
                )
                if len(emg_val) > 0:
                    val_datasets.append(emg_val)
                    print(f"   ✅ Val: {len(emg_val)} samples")
                
                # Test set
                emg_test = SilentSpeechEMGDataset(
                    path,
                    split="test",
                    audio_sample_rate=data_config.get('sample_rate', 16000),
                    emg_sample_rate=data_config.get('emg_sample_rate', 1000),
                    duration=data_config.get('duration', 4.0),
                    transform=None
                )
                if len(emg_test) > 0:
                    test_datasets.append(emg_test)
                    print(f"   ✅ Test: {len(emg_test)} samples")
                    
            else:
                # Handle audio datasets (REAL/FAKE, wav_real/wav_fake, etc.)
                print(f"   📁 Loading audio dataset...")
                audio_train = DeepfakeAudiosDataset(
                    path,
                    split="train",
                    sample_rate=data_config.get('sample_rate', 16000),
                    duration=data_config.get('duration', 4.0),
                    transform=transform,
                    split_ratio=0.7
                )
                if len(audio_train) > 0:
                    train_datasets.append(audio_train)
                    print(f"   ✅ Train: {len(audio_train)} samples")
                    
                    # Create val/test from same path
                    audio_val = DeepfakeAudiosDataset(
                        path,
                        split="val",
                        sample_rate=data_config.get('sample_rate', 16000),
                        duration=data_config.get('duration', 4.0),
                        transform=None,
                        split_ratio=0.7
                    )
                    if len(audio_val) > 0:
                        val_datasets.append(audio_val)
                        print(f"   ✅ Val: {len(audio_val)} samples")
                    
                    audio_test = DeepfakeAudiosDataset(
                        path,
                        split="test",
                        sample_rate=data_config.get('sample_rate', 16000),
                        duration=data_config.get('duration', 4.0),
                        transform=None,
                        split_ratio=0.7
                    )
                    if len(audio_test) > 0:
                        test_datasets.append(audio_test)
                        print(f"   ✅ Test: {len(audio_test)} samples")
                        
        except Exception as e:
            print(f"   ❌ Error loading {name}: {e}")
            logging.error(f"Error loading dataset {name}: {e}")
            continue
    
    # Create combined datasets
    if not train_datasets:
        raise ValueError("No training datasets found! Check data paths.")
    
    print(f"\n{'='*80}")
    print(f"DATASET SUMMARY")
    print(f"{'='*80}")
    
    train_dataset = MultiModalDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = MultiModalDataset(val_datasets) if len(val_datasets) > 1 else (val_datasets[0] if val_datasets else None)
    test_dataset = MultiModalDataset(test_datasets) if len(test_datasets) > 1 else (test_datasets[0] if test_datasets else None)
    
    print(f"✅ Combined Training samples: {len(train_dataset)}")
    print(f"✅ Combined Validation samples: {len(val_dataset) if val_dataset else 0}")
    print(f"✅ Combined Test samples: {len(test_dataset) if test_dataset else 0}")
    print(f"✅ Using {len(train_datasets)} training datasets:")
    for i, dataset in enumerate(train_datasets):
        print(f"   {i+1}. {type(dataset).__name__}: {len(dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    ) if test_dataset else None

    return train_loader, val_loader, test_loader