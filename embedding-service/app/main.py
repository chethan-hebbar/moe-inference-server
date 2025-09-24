# embedding-service/app/main.py
import torch
import torch.nn as nn
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
from contextlib import asynccontextmanager

# We need the PositionalEncoding class
from model import PositionalEncoding

# --- Configuration ---
VOCAB_SIZE = 95812 # Must match your trained vocab size
D_MODEL = 128
MODEL_PATH = Path(__file__).resolve().parent / "embedding_layer.pth"

# --- Pydantic Models ---
class EmbeddingRequest(BaseModel):
    token_indices: List[int]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

# --- Lifespan and App ---
_dependencies = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Embedding Service: Loading dependencies...")
    deps = {}
    if not MODEL_PATH.exists():
        print(f"Embedding Service: Error - Weights not found at {MODEL_PATH}")
    else:
        try:
            embedding_layer = nn.Embedding(VOCAB_SIZE, D_MODEL)
            embedding_layer.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            embedding_layer.eval()
            deps["embedding_layer"] = embedding_layer
            
            # Positional encoding is not learned, just instantiated
            deps["pos_encoder"] = PositionalEncoding(D_MODEL)
            
            print("Embedding Service: Trained embedding layer loaded successfully.")
        except Exception as e:
            print(f"Embedding Service: Error loading model: {e}")
    
    app.state.dependencies = deps
    yield
    app.state.dependencies.clear()

app = FastAPI(title="Embedding Service", lifespan=lifespan)

@app.post("/embed", response_model=EmbeddingResponse)
def get_embeddings(request: EmbeddingRequest):
    deps = app.state.dependencies
    if "embedding_layer" not in deps or "pos_encoder" not in deps:
        raise HTTPException(status_code=500, detail="Embedding models not loaded.")
    
    indices_tensor = torch.tensor(request.token_indices, dtype=torch.long).unsqueeze(0) # Add batch dim
    
    with torch.no_grad():
        # Apply embedding, scaling, and positional encoding
        embedded = deps["embedding_layer"](indices_tensor) * math.sqrt(D_MODEL)
        final_embeddings = deps["pos_encoder"](embedded)
    
    output_list = final_embeddings.squeeze(0).tolist()
    return EmbeddingResponse(embeddings=output_list)