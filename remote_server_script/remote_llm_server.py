# Remote_llm_server.py
# -*- coding: utf-8 -*-
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_server import ModelConfig, LLMInterface, LoRALLMInterface

app = FastAPI(title="Remote LLM Server")

_cfg = ModelConfig()
_base_llm: Optional[LLMInterface] = None
_lora_llm: Optional[LoRALLMInterface] = None


def get_base_llm() -> LLMInterface:
    global _base_llm
    if _base_llm is None:
        logging.info("Loading BASE LLM...")
        _base_llm = LLMInterface(_cfg)
    return _base_llm


def get_lora_llm() -> LoRALLMInterface:
    global _lora_llm
    if _lora_llm is None:
        logging.info("Loading LoRA LLM...")
        _lora_llm = LoRALLMInterface(_cfg)
    return _lora_llm


def unload_base_llm():
    global _base_llm
    if _base_llm is not None:
        try:
            _base_llm.unload()
        except Exception as e:
            logging.warning(f"unload_base_llm: {e}")
        _base_llm = None
        logging.info("Base LLM unloaded via API.")


def unload_lora_llm():
    global _lora_llm
    if _lora_llm is not None:
        try:
            _lora_llm.unload()
        except Exception as e:
            logging.warning(f"unload_lora_llm: {e}")
        _lora_llm = None
        logging.info("LoRA LLM unloaded via API.")


class GenerateRequest(BaseModel):
    prompt: str
    use_lora: bool = False 


class GenerateResponse(BaseModel):
    output: str


class CountTokensRequest(BaseModel):
    text: str
    use_lora: bool = False
    add_special_tokens: bool = False


class CountTokensResponse(BaseModel):
    tokens: int


class UnloadRequest(BaseModel):
    use_lora: bool = False


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.use_lora:
        llm = get_lora_llm()
    else:
        llm = get_base_llm()

    text = llm.generate(req.prompt)
    return GenerateResponse(output=text)


@app.post("/count_tokens", response_model=CountTokensResponse)
def count_tokens(req: CountTokensRequest):
    if req.use_lora:
        llm = get_lora_llm()
    else:
        llm = get_base_llm()

    tok = llm.get_tokenizer()
    if tok is None:
        raise HTTPException(status_code=500, detail="Tokenizer not available")

    try:
        ids = tok.encode(req.text, add_special_tokens=req.add_special_tokens)
        return CountTokensResponse(tokens=len(ids))
    except Exception as e:
        logging.exception(f"Tokenization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
def unload(req: UnloadRequest):
    if req.use_lora:
        unload_lora_llm()
    else:
        unload_base_llm()
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    uvicorn.run("remote_llm_server:app", host="0.0.0.0", port=8000)
