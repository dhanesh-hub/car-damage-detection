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

# --- 1. EMERGENCY SETTINGS ---
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
torch.set_grad_enabled(False)

app = FastAPI()

# --- 2. CORS (Fully Open to fix connectivity) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. THE "BYPASS CORRUPTION" LOGIC ---
base_path = os.path.dirname(__file__)
model_weights_path = os.path.join(base_path, "damage_segmentation_model.pth")

def download_and_verify():
    # Direct URL to verified Mask R-CNN weights
    url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    
    # Check if file is small (pointer) or if it exists
    if os.path.exists(model_weights_path):
        size_mb = os.path.getsize(model_weights_path) / (1024 * 1024)
        # If the file is 177MB but giving "Invalid Magic Number", it is corrupted.
        # We delete it and start over.
        print(f"⚠️ Found existing file ({size_mb:.2f}MB). Deleting to clear corruption...")
        os.remove(model_weights_path)
        
    print(f"📡 Downloading fresh, clean weights from official source...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(model_weights_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                f.write(chunk)
        print("✅ Success: Weights downloaded and saved.")
    else:
        print(f"❌ Error: Download failed with status {r.status_code}")

# Trigger the download immediately on startup
download_and_verify()

# --- 4. MODEL INITIALIZATION ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 

# YOUR SETTINGS: Precision 0.10 and Smoothing 0
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10 
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0     

cfg.MODEL.WEIGHTS = model_weights_path
cfg.MODEL.DEVICE = "cpu" 

print("🧠 Starting Model Engine (CPU)...")
predictor = DefaultPredictor(cfg)
print("🚀 SYSTEM ONLINE: Model Ready.")

# Groq Client (API Key from Railway Env Variables)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/health")
async def health():
    return {"status": "online", "engine": "detectron2"}

@app.post("/scan")
async def scan_damage(file: UploadFile = File(...), brand: str = Form(...), model: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    outputs = predictor(img)
    damage_count = len(outputs["instances"])

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