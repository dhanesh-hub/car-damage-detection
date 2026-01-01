import os
import io
import base64
import gc
import cv2
import numpy as np
import torch
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# --- 1. EMERGENCY FIXES FOR DEPLOYMENT ---
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
torch.set_grad_enabled(False)

app = FastAPI()

# --- 2. CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "https://car-damage-detection-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. MODEL WEIGHTS FALLBACK SYSTEM ---
base_path = os.path.dirname(__file__)
model_weights_path = os.path.join(base_path, "damage_segmentation_model.pth")

# URL to a known direct download (Using a standard Detectron2 COCO model as fallback)
# If you have your own Dropbox/Drive direct link, replace this URL:
FALLBACK_URL = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

def verify_weights():
    """Checks if weights exist and are not just a Git LFS pointer."""
    if os.path.exists(model_weights_path):
        file_size = os.path.getsize(model_weights_path)
        if file_size > 1000000: # If larger than 1MB, it's likely real data
            print(f"✅ Valid weights found locally ({file_size / 1e6:.2f} MB)")
            return
    
    print("⚠️ Local weights missing or corrupted pointer detected. Downloading fallback...")
    response = requests.get(FALLBACK_URL, stream=True)
    with open(model_weights_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Download complete.")

verify_weights()

# --- 4. AI MODEL INITIALIZATION ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 

# PREFERENCE: Precision 0.10 and Smoothness 0
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10 
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0     

cfg.MODEL.WEIGHTS = model_weights_path
cfg.MODEL.DEVICE = "cpu" 

# Initialize Predictor
predictor = DefaultPredictor(cfg)

# Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": os.path.exists(model_weights_path)}

@app.post("/scan")
async def scan_damage(file: UploadFile = File(...), brand: str = Form(...), model: str = Form(...)):
    # 1. Read and Decode Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run AI Inference
    outputs = predictor(img)
    damage_count = len(outputs["instances"])

    # 3. Generate Billing Logic with Groq
    img_b64 = base64.b64encode(contents).decode('utf-8')
    prompt = f"Estimate repair for a {brand} {model}. Found {damage_count} damage spots. Provide a structured HTML table with INR costs."
    
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}]
    )

    gc.collect()

    return {
        "damage_found": damage_count,
        "bill_html": response.choices[0].message.content
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)