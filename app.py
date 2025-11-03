import os
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_ID = os.environ.get("MODEL_ID", "TesterColab/mistral-finetunedv2")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Load tokenizer and model once when container starts
print(f"Loading model {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, resume_download=True, trust_remote_code=True, use_safetensors=True, device_map="auto", token=HF_TOKEN)
# if device_map="auto" errors, remove it and call .to("cuda") after loading

@app.post("/predict")
async def predict(req: Request):
    data = await req.json()
    prompt = data.get("input", "")
    if not prompt:
        return {"error": "no input provided"}
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"output": text}
