from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# -----------------------------
# Load environment variables
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME")

if not HF_TOKEN or not MODEL_NAME:
    raise ValueError("HF_TOKEN or MODEL_NAME environment variable not set!")

# -----------------------------
# Load tokenizer and model
# -----------------------------
print(f"Loading model {MODEL_NAME} from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model.eval()  # set model to evaluation mode

print("Model loaded successfully!")

# -----------------------------
# Setup FastAPI
# -----------------------------
app = FastAPI(title="Mistral 7B API")

class RequestBody(BaseModel):
    prompt: str
    max_length: int = 150

@app.post("/generate")
def generate_text(body: RequestBody):
    inputs = tokenizer(body.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=body.max_length)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}

@app.get("/")
def root():
    return {"message": "Mistral 7B API is running!"}
