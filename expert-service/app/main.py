# expert-service/app/main.py
import os
import torch
import torch.nn as nn
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
from contextlib import asynccontextmanager

from expert import Expert
# We need the full model definition to create the embedding layer
from model import MoETransformerClassifier

# --- 1. Configuration ---
D_MODEL = 128
D_HIDDEN = 512
# This vocab size must match the one from training
VOCAB_SIZE = 95811 
EXPERT_ID = os.environ.get("EXPERT_ID")
MODEL_PATH = Path(__file__).resolve().parent / f"expert_{EXPERT_ID}.pth" if EXPERT_ID else None

# --- 2. Pydantic Models ---
class ExpertRequest(BaseModel):
    embeddings: List[List[float]]

class ExpertResponse(BaseModel):
    processed_representations: List[List[float]]

# --- 3. App, Models, and Embeddings ---
_dependencies = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print(f"Expert Service {EXPERT_ID}: Loading dependencies...")

    if MODEL_PATH is not None and MODEL_PATH.exists():
        try:
            expert_model = Expert(d_model=D_MODEL, d_hidden=D_HIDDEN)
            expert_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            expert_model.eval()
            _dependencies["expert_model"] = expert_model
            print(f"Expert {EXPERT_ID}: Model loaded successfully.")
        except Exception as e:
            print(f"Expert {EXPERT_ID}: Error loading model: {e}")
    
    app.state.dependencies = _dependencies
    yield
    # --- Shutdown ---
    app.state.dependencies.clear()
    print(f"Expert Service {EXPERT_ID}: Shutting down.")


app = FastAPI(title=f"MoE Expert Service (ID: {EXPERT_ID})", lifespan=lifespan)

@app.post("/process", response_model=ExpertResponse)
def process_tokens(request: ExpertRequest):
    deps = app.state.dependencies
    if "expert_model" not in deps:
        raise HTTPException(status_code=500, detail=f"Model for Expert {EXPERT_ID} not loaded.")
    
    input_repr = torch.tensor(request.embeddings).unsqueeze(0) # Add batch dim

    # 2. Perform a forward pass
    with torch.no_grad():
        processed_repr = deps["expert_model"](input_repr)
    
    output_list = processed_repr.squeeze(0).tolist()
    print(f"Expert {EXPERT_ID} successfully processed {len(output_list)} tokens.")
    return ExpertResponse(processed_representations=output_list)