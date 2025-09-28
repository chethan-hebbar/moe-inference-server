# main.py
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
from contextlib import asynccontextmanager
import torch.nn.functional as F

# Import the Gating class from our gating.py file
from gating import Gating
from dataloader import get_tokenizer_and_vocab

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

# --- 1. Configuration ---
# These parameters must match the model you trained in Phase 1
D_MODEL = 128
NUM_EXPERTS = 8
TOP_K = 2
MODEL_PATH = Path(__file__).resolve().parent / "gating_network.pth"

# --- 2. Pydantic Models for Request and Response ---
# These define the JSON structure for our API
class GatingRequest(BaseModel):
    embeddings: List[List[float]]

class GatingResponse(BaseModel):
    # This is the routing decision
    top_k_indices: List[int]
    top_k_weights: List[float]

class TokenizeRequest(BaseModel):
    text: str

class TokenizeResponse(BaseModel):
    token_indices: List[int]

# --- 4. Load the Model ---
# This block of code will run once when the application starts.
model = None
tokenizer = None
vocab = None

_dependencies = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code to run on startup ---
    print("Gating Service: Loading dependencies...")
    try:
        tokenizer, vocab = get_tokenizer_and_vocab()
        _dependencies["tokenizer"] = tokenizer
        _dependencies["vocab"] = vocab
        print("Gating Service: Tokenizer and vocab loaded successfully.")
    except Exception as e:
        print(f"Gating Service: Error loading tokenizer/vocab: {e}")
    
    if not MODEL_PATH.exists():
        print(f"Gating Service: Error - Model weights not found at {MODEL_PATH}")
    else:
        try:
            model = Gating(d_model=D_MODEL, num_experts=NUM_EXPERTS)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            _dependencies["model"] = model
            print("Gating Service: Gating network model loaded successfully.")
        except Exception as e:
            print(f"Gating Service: Error loading model: {e}")
    
    # Assign the populated dictionary to app.state
    app.state.dependencies = _dependencies
    yield
    # --- Code to run on shutdown ---
    _dependencies.clear()
    print("Gating Service: Shutting down and cleaning up resources.")


app = FastAPI(title="MoE Gating Service", lifespan=lifespan)


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize_text(request: TokenizeRequest):
    deps = app.state.dependencies
    if "tokenizer" not in deps or "vocab" not in deps:
        raise HTTPException(status_code=500, detail="Tokenizer/vocab not loaded.")
    
    tokens = deps["tokenizer"](request.text)
    indices = deps["vocab"](tokens)
    return TokenizeResponse(token_indices=indices)


@app.post("/route", response_model=GatingResponse)
def get_routing_decision(request: GatingRequest):
    deps = app.state.dependencies
    if "model" not in deps:
        raise HTTPException(status_code=500, detail="Gating model not loaded.")

    embedding_tensor = torch.tensor(request.embeddings).unsqueeze(0) # Add batch dim

    # 2. Perform a forward pass
    with torch.no_grad():
        gating_logits = deps["model"](embedding_tensor)

    # 3. Get weights and indices from the first token's logits
    first_token_logits = gating_logits[0, 0, :]
    top_k_weights_tensor, top_k_indices = torch.topk(first_token_logits, TOP_K)
    top_k_weights = F.softmax(top_k_weights_tensor, dim=-1)
    
    indices_list = top_k_indices.tolist()
    weights_list = top_k_weights.tolist()
    
    print(f"Routing request to experts {indices_list} with weights {weights_list}")
    return GatingResponse(top_k_indices=indices_list, top_k_weights=weights_list)