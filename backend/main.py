import io
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
from groq import Groq
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
import gc

# 1. Disable Gradient Calculation (saves ~30% memory)
torch.set_grad_enabled(False)

# 2. Force CPU-only mode (prevents CUDA initialization overhead)
device = torch.device("cpu")

# 3. Add this after your model finishes a prediction
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache() # Even on CPU, this helps clear PyTorch's internal allocator

app = FastAPI()

# --- CORS SETUP: Allow React Frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI MODEL INITIALIZATION ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10 # Your Precision Requirement
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0     # Your Smoothing Requirement
# Example using a direct download URL
cfg.MODEL.WEIGHTS = "https://your-storage-link.com/damage_segmentation_model.pth"
cfg.MODEL.DEVICE = "cpu" # Change to "cuda" if using GPU
predictor = DefaultPredictor(cfg)
client = Groq(api_key="YOUR_GROQ_API_KEY")

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

    return {
        "damage_found": damage_count,
        "bill_html": response.choices[0].message.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)