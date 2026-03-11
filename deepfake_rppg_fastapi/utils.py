import torch
import torch.nn as nn
from torchvision import models
import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
import os

# --- Model Architecture ---

class RPPGResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(RPPGResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        original_first_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            self.resnet.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
            
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

# --- Face Detection ---

class FaceDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def get_roi_coordinates(self, landmarks, frame_shape):
        h, w, _ = frame_shape
        fh_indices = [68, 104, 69, 108, 151, 337, 299, 333, 298]
        lc_indices = [205, 207, 214, 212, 165, 92, 186, 57, 43, 106, 182]
        rc_indices = [373, 374, 280, 425, 427, 411, 287, 273, 422, 391, 326]

        rois = {
            'forehead': self._get_mask(fh_indices, landmarks, h, w),
            'left_cheek': self._get_mask(lc_indices, landmarks, h, w),
            'right_cheek': self._get_mask(rc_indices, landmarks, h, w)
        }
        return rois

    def _get_mask(self, indices, landmarks, h, w):
        points = []
        for idx in indices:
            pt = landmarks.landmark[idx]
            points.append([int(pt.x * w), int(pt.y * h)])
        points = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        return mask

# --- Signal Processing ---

def normalize_signal(sig):
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

def filter_signal(sig, fs=30, low=0.7, high=4.0):
    b, a = signal.butter(3, [low, high], btype='bandpass', fs=fs)
    return signal.filtfilt(b, a, sig)

def generate_spectrogram(sig, fs=30, nperseg=128, noverlap=100):
    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx + 1e-12)
    Sxx_min = np.min(Sxx)
    Sxx_max = np.max(Sxx)
    if Sxx_max > Sxx_min:
        Sxx = (Sxx - Sxx_min) / (Sxx_max - Sxx_min)
    return Sxx

def get_rppg_signal(raw_traces, method='pos'):
    traces = np.array(raw_traces)
    if method == 'pos':
        win_size = 32
        n_frames = len(traces)
        H = np.zeros(n_frames)
        for i in range(n_frames - win_size):
            C = traces[i:i+win_size, :].T
            mean_C = np.mean(C, axis=1, keepdims=True)
            C_norm = C / (mean_C + 1e-6)
            S = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
            P = S @ C_norm
            std_p1 = np.std(P[0, :])
            std_p2 = np.std(P[1, :])
            alpha = std_p1 / (std_p2 + 1e-6)
            h = P[0, :] + alpha * P[1, :]
            H[i:i+win_size] += (h - np.mean(h))
        return H
    return traces[:, 1]

def extract_raw_trace(frames, detector: FaceDetector):
    traces = {'forehead': [], 'left_cheek': [], 'right_cheek': []}
    for frame in frames:
        landmarks = detector.process_frame(frame)
        if landmarks is None:
            for key in traces: traces[key].append([0, 0, 0])
            continue
        masks = detector.get_roi_coordinates(landmarks, frame.shape)
        for region_name, mask in masks.items():
            mean_color = cv2.mean(frame, mask=mask)[:3]
            traces[region_name].append(mean_color[::-1])
    return traces

# --- Inference ---

def predict_video(video_path, model, detector, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    MAX_FRAMES = 300
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if len(frames) < 30:
        return None, "Video too short"

    traces = extract_raw_trace(frames, detector)
    signals = []
    for key in traces:
        if len(traces[key]) > 0:
            sig = get_rppg_signal(traces[key], method='pos')
            signals.append(sig)
    
    if not signals:
        return None, "No face detected"

    final_signal = np.mean(signals, axis=0)
    norm_sig = normalize_signal(final_signal)
    filt_sig = filter_signal(norm_sig)
    spectrogram = generate_spectrogram(filt_sig)
    spec_resized = cv2.resize(spectrogram, (224, 224))
    spec_tensor = torch.tensor(spec_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(spec_tensor)
        prob = output.item()
    
    label = "FAKE" if prob > 0.5 else "REAL"
    return {"label": label, "probability": prob}, None
