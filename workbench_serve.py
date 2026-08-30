"""GPU Workbench service for two Gemma + PEFT Iris adapters.

Run on the Workbench VM after copying the adapter folders from Cloud Storage:
  V1_ADAPTER_DIR=... V2_ADAPTER_DIR=... uvicorn workbench_serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_guardrails import GuardedIrisPipeline

BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-3-1b-it")
ADAPTER_PATHS = {"v1": os.getenv("V1_ADAPTER_DIR", "models/v1_adapter"),
                 "v2": os.getenv("V2_ADAPTER_DIR", "models/v2_adapter")}
models: dict[str, tuple[AutoTokenizer, PeftModel]] = {}


def load_adapter(adapter_path: str) -> tuple[AutoTokenizer, PeftModel]:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


def model_predict(prompt: str, version: str) -> str:
    tokenizer, model = models[version]
    # Match the prompt prefix used by the checked-in v1 fine-tuning notebook.
    prefix = "Classify the following Iris flower.\n\n"
    suffix = "\n\nSpecies:"
    inputs = tokenizer(prefix + prompt + suffix, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for version, adapter_path in ADAPTER_PATHS.items():
        if not os.path.isdir(adapter_path):
            raise RuntimeError(f"{version} adapter folder is missing: {adapter_path}")
        models[version] = load_adapter(adapter_path)
    yield
    models.clear()


app = FastAPI(title="Guarded Iris Gemma service", lifespan=lifespan)
guarded = GuardedIrisPipeline(model_predict)


class PromptRequest(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"loaded_versions": sorted(models)}


@app.post("/{version}/predict")
def predict(version: str, request: PromptRequest):
    if version not in ADAPTER_PATHS:
        raise HTTPException(status_code=404, detail="version must be v1 or v2")
    return guarded.predict(request.prompt, version)


@app.post("/{version}/raw_predict", include_in_schema=False)
def raw_predict_for_isolated_evaluation(version: str, request: PromptRequest):
    """Baseline endpoint for the assignment only; never expose it publicly."""
    if version not in ADAPTER_PATHS:
        raise HTTPException(status_code=404, detail="version must be v1 or v2")
    return {"response": model_predict(request.prompt, version)}
