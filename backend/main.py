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

# --- 1. EMERGENCY FIXES ---
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
torch.set_grad_enabled(False)

app = FastAPI()

# --- 2. CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Temporarily open to everything to rule out CORS issues
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. WEIGHTS RECOVERY LOGIC ---
base_path = os.path.dirname(__file__)
model_weights_path = os.path.join(base_path, "damage_segmentation_model.pth")

def download_fresh_weights():
    # Official Detectron2 Weights URL (Mask R-CNN R50-FPN 3x)
    url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    
    if os.path.exists(model_weights_path):
        print(f"🗑️ Found corrupted file. Deleting...")
        os.remove(model_weights_path)
        
    print(f"📡 Downloading fresh weights from {url}...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(model_weights_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Weights downloaded successfully.")
    else:
        print(f"❌ Download failed! Status: {r.status_code}")

# Run the download
download_fresh_weights()

# --- 4. MODEL CONFIG ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 

# PREFERENCE: Precision 0.10 and Smoothness 0
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10 
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0     

cfg.MODEL.WEIGHTS = model_weights_path
cfg.MODEL.DEVICE = "cpu" 

print("🧠 Initializing Predictor (This may take a moment)...")
predictor = DefaultPredictor(cfg)
print("🚀 Model Loaded Successfully!")

# Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/health")
async def health():
    return {"status": "online", "model": "loaded"}

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