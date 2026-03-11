from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import torch
from utils import RPPGResNet, FaceDetector, predict_video
import uvicorn

app = FastAPI(title="rPPG Deepfake Detector API")

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RPPGResNet()
model_path = 'rppg_model.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    print(f"Warning: {model_path} not found. Prediction will use random weights.")

model.to(device)
model.eval()

detector = FaceDetector()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result, error = predict_video(temp_path, model, detector, device)
        if error:
            raise HTTPException(status_code=400, detail=error)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def root():
    return {"message": "rPPG Deepfake Detector API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
