# full-model-service/app/main.py
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# Import from the common library (assuming it's in the PYTHONPATH)
from model import MoETransformerClassifier
from dataloader import get_tokenizer_and_vocab
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

# --- 1. Configuration ---
# These MUST match the parameters of your best trained model
VOCAB_SIZE = 95812
D_MODEL = 128
NHEAD = 4
D_HIDDEN = 512
NUM_EXPERTS = 8
TOP_K = 2
NUM_LAYERS = 2
NUM_CLASSES = 4
MODEL_PATH = "full_model.pth"
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

# --- 2. Pydantic Models for the API ---
class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    predicted_category: str
    confidence: float

# --- 3. Lifespan Manager for Model Loading ---
_dependencies = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Full Model Service: Loading dependencies...")
    deps = {}
    try:
        # Load the tokenizer and vocab
        deps["tokenizer"], deps["vocab"] = get_tokenizer_and_vocab()
        print("Full Model Service: Tokenizer and vocab loaded successfully.")
        
        # Define model architecture
        model_params = {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'nhead': NHEAD,
            'd_hidden': D_HIDDEN,
            'num_experts': NUM_EXPERTS,
            'top_k': TOP_K,
            'num_classes': NUM_CLASSES,
            'num_layers': NUM_LAYERS
        }
        
        # Load the FULL trained model
        model = MoETransformerClassifier(**model_params)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        deps["model"] = model
        print(f"Full Model Service: Model loaded successfully from {MODEL_PATH}.")

    except Exception as e:
        print(f"Full Model Service: An error occurred during startup: {e}")

    app.state.dependencies = deps
    yield
    app.state.dependencies.clear()


app = FastAPI(title="Full MoE AI Service", lifespan=lifespan)


@app.post("/infer", response_model=InferenceResponse)
def perform_inference(request: InferenceRequest):
    deps = app.state.dependencies
    if "model" not in deps or "tokenizer" not in deps or "vocab" not in deps:
        raise HTTPException(status_code=500, detail="Model or dependencies not loaded.")

    tokenizer = deps["tokenizer"]
    vocab = deps["vocab"]
    model = deps["model"]

    # 1. Pre-process: Tokenize and numericalize the input text
    tokens = tokenizer(request.text)
    indices = vocab(tokens)
    indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # 2. Full Model Inference
    with torch.no_grad():
        # The model's forward pass returns final logits and the list of gating logits
        output_logits, _ = model(indices_tensor)

    # 3. Post-process: Interpret the results
    probabilities = F.softmax(output_logits, dim=-1)
    confidence, predicted_index_tensor = torch.max(probabilities, dim=1)

    predicted_index = predicted_index_tensor.item()
    predicted_category = CATEGORIES[predicted_index]
    
    print(f"Prediction for '{request.text[:30]}...': {predicted_category} ({confidence.item():.4f})")

    return InferenceResponse(
        predicted_category=predicted_category,
        confidence=confidence.item()
    )