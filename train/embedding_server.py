import argparse
import os
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


app = FastAPI()
MODEL: SentenceTransformer | None = None


class EncodeRequest(BaseModel):
    texts: List[str]


class EncodeResponse(BaseModel):
    embeddings: List[List[float]]


@app.post("/encode", response_model=EncodeResponse)
def encode_texts(request: EncodeRequest) -> EncodeResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Encoder not initialised")
    if not request.texts:
        return EncodeResponse(embeddings=[])

    embeddings = MODEL.encode(
        request.texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return EncodeResponse(embeddings=embeddings.tolist())


def build_app(model_path: str, device: str) -> FastAPI:
    global MODEL
    MODEL = SentenceTransformer(model_path, device=device)
    try:
        MODEL = MODEL.to(device)
    except AttributeError:
        pass
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SentenceTransformer encoding service")
    parser.add_argument("--model_path", type=str, required=True, help="Path or name of the encoder model")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--uvicorn-log-level", type=str, default="info")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    build_app(args.model_path, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.uvicorn_log_level)


if __name__ == "__main__":
    main()
