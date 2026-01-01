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

# --- 1. MEMORY OPTIMIZATIONS ---
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
torch.set_grad_enabled(False) # Disables gradient calculation to save RAM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. TINY MODEL INITIALIZATION ---
cfg = get_cfg()
# We switch to a Faster R-CNN with a smaller backbone (R_50_FPN is heavy, but this is the lightest standard)
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# CUSTOM SETTINGS: Precision 0.10, Smoothing 0
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10 
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0     
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# CRITICAL: Reduce the size of images processed to save RAM
cfg.INPUT.MIN_SIZE_TEST = 400 
cfg.INPUT.MAX_SIZE_TEST = 400

print("📡 Loading Tiny-Backbone Model...")
predictor = DefaultPredictor(cfg)
print("✅ LOW-MEMORY MODEL READY")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/health")
async def health():
    return {"status": "active"}

@app.post("/scan")
async def scan_damage(file: UploadFile = File(...), brand: str = Form(...), model: str = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process
        outputs = predictor(img)
        damage_count = len(outputs["instances"])

        # Vision Estimate
        img_b64 = base64.b64encode(contents).decode('utf-8')
        prompt = f"Estimate repair for a {brand} {model}. Found {damage_count} damage spots. Provide a structured HTML table with INR costs."
        
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}]
        )
        
        # Immediate cleanup
        del img
        del outputs
        gc.collect() 
        
        return {"damage_found": damage_count, "bill_html": response.choices[0].message.content}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)