# classifier-service/app/main.py
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
from contextlib import asynccontextmanager

# --- Configuration ---
D_MODEL = 128
NUM_CLASSES = 4 # AG News
MODEL_PATH = Path(__file__).resolve().parent / "classifier_head.pth"

# --- Pydantic Models ---
class ClassifierRequest(BaseModel):
    pooled_vector: List[float]

class ClassifierResponse(BaseModel):
    logits: List[float]

# --- Lifespan and App ---
_dependencies = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Classifier Service: Loading dependencies...")
    deps = {}
    if not MODEL_PATH.exists():
        print(f"Classifier Service: Error - Weights not found at {MODEL_PATH}")
    else:
        try:
            model = nn.Linear(D_MODEL, NUM_CLASSES)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            deps["classifier_model"] = model
            print("Classifier Service: Trained classifier head loaded successfully.")
        except Exception as e:
            print(f"Classifier Service: Error loading model: {e}")
    
    app.state.dependencies = deps
    yield
    app.state.dependencies.clear()

app = FastAPI(title="Classifier Service", lifespan=lifespan)

@app.post("/classify", response_model=ClassifierResponse)
def get_classification(request: ClassifierRequest):
    deps = app.state.dependencies
    if "classifier_model" not in deps:
        raise HTTPException(status_code=500, detail="Classifier model not loaded.")
    
    # Convert input list to a tensor
    pooled_tensor = torch.tensor(request.pooled_vector).unsqueeze(0) # Add batch dim
    
    with torch.no_grad():
        # Perform the final classification
        output_logits = deps["classifier_model"](pooled_tensor)
    
    output_list = output_logits.squeeze(0).tolist() # Remove batch dim
    return ClassifierResponse(logits=output_list)