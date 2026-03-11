import torch
import numpy as np
import yaml
import sys
import librosa
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.fusion_model import create_model
from features import AudioFeatureExtractor

# Load config
with open('config_fixed.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=== TESTING CORRECTED MODEL ===")

# Load corrected model
model = create_model(config)
checkpoint = torch.load('outputs_corrected/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best validation accuracy: {checkpoint.get('best_accuracy', 'N/A')}")

# Create feature extractor
feature_extractor = AudioFeatureExtractor(config)
feature_extractor.eval()

def test_audio_file(file_path):
    """Test a single audio file"""
    print(f"\n=== TESTING {file_path} ===")
    
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=config['data']['sample_rate'])
        target_length = int(config['data']['sample_rate'] * config['data']['duration'])
        
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            features = feature_extractor(audio_tensor)
            feature_tensor = feature_extractor.get_feature_tensor(features)
            outputs = model(feature_tensor)
            probabilities = outputs['probabilities']
            
            fake_prob = probabilities[0].item()
            real_prob = 1 - fake_prob
            
            print(f"Raw logits: {outputs['logits']}")
            print(f"Probabilities: {probabilities}")
            print(f"Fake prob: {fake_prob:.6f}, Real prob: {real_prob:.6f}")
            
            # CORRECTED LOGIC: FAKE=1, REAL=0
            if fake_prob > 0.5:
                prediction = "FAKE"
                confidence = fake_prob * 100
            else:
                prediction = "REAL"
                confidence = real_prob * 100
            
            print(f"🎵 PREDICTION: {prediction}")
            print(f"📊 CONFIDENCE: {confidence:.1f}%")
            
            return prediction, confidence, fake_prob, real_prob
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None, None, None, None

# Test both files
fake_pred, fake_conf, fake_fp, fake_rp = test_audio_file("data/fake.wav")
real_pred, real_conf, real_fp, real_rp = test_audio_file("data/real.wav")

print("\n=== SUMMARY ===")
print(f"fake.wav → {fake_pred} ({fake_conf:.1f}% confidence)")
print(f"real.wav → {real_pred} ({real_conf:.1f}% confidence)")

if fake_pred and real_pred:
    if fake_pred == "FAKE" and real_pred == "REAL":
        print("✅ MODEL WORKS PERFECTLY!")
    elif fake_pred == "REAL" and real_pred == "FAKE":
        print("⚠️ MODEL IS INVERTED (swapped predictions)")
    elif fake_pred == real_pred:
        print("❌ MODEL PREDICTS SAME FOR BOTH FILES")
    else:
        print("🤔 MODEL HAS MIXED RESULTS")
