import os
import io
import base64
import gc
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# --- 1. SYSTEM & MEMORY FIXES ---
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
torch.set_grad_enabled(False)

app = FastAPI()

# --- 2. EXPLICIT CORS CONFIGURATION ---
# This fixes the "No 'Access-Control-Allow-Origin' header" error
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your local frontend (localhost:5173) to connect
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --- 3. THE STABLE INITIALIZATION ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# PREFERENCES: Precision 0.10, Smoothing 0
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10 
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0     
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
cfg.MODEL.DEVICE = "cpu"

# Force download from official Model Zoo to avoid "Magic Number" corruption
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

print("📡 System is downloading/loading weights from Model Zoo...")
predictor = DefaultPredictor(cfg)
print("✅ MODEL LOADED SUCCESSFULLY")

# Groq Client (Ensure GROQ_API_KEY is in your Railway Variables)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/health")
async def health():
    return {"status": "active"}

@app.post("/scan")
async def scan_damage(file: UploadFile = File(...), brand: str = Form(...), model: str = Form(...)):
    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Run Detectron2 Prediction
        outputs = predictor(img)
        damage_count = len(outputs["instances"])

        # 2. Prepare Image for Groq Vision
        img_b64 = base64.b64encode(contents).decode('utf-8')
        
        # 3. Request Repair Estimate from Llama-3-Vision
        prompt = (
            f"Estimate repair for a {brand} {model}. "
            f"The AI detected {damage_count} damage locations. "
            "Provide a professional, structured HTML table with estimated INR costs for labor and parts."
        )
        
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }]
        )

        # Clear memory after heavy processing
        gc.collect()

        return {
            "damage_found": damage_count, 
            "bill_html": response.choices[0].message.content
        }
    except Exception as e:
        print(f"❌ ERROR DURING SCAN: {str(e)}")
        return {"error": "Processing failed", "details": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)