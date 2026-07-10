import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.rag import build_rag_chain
from app.retrieval_pipeline import get_vector_store
from app.voice import transcribe, synthesize

_rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_chain
    vector_store = get_vector_store()
    _rag_chain = build_rag_chain(vector_store)
    yield


app = FastAPI(
    title="YAITEC RAG Assistant",
    description="Assistente RAG sobre a base de conhecimento do YAITEC Atende.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class AskRequest(BaseModel):
    pergunta: str


class AskResponse(BaseModel):
    resposta: str
    fontes: list[str]


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    if not body.pergunta.strip():
        raise HTTPException(status_code=422, detail="pergunta não pode ser vazia")
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain não inicializada")
    result = _rag_chain(body.pergunta)
    return AskResponse(resposta=result["resposta"], fontes=result["fontes"])


class AskAudioResponse(BaseModel):
    pergunta: str
    resposta: str
    fontes: list[str]
    audio_base64: str


@app.post("/ask-audio", response_model=AskAudioResponse)
async def ask_audio(file: UploadFile = File(...)):
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain não inicializada")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="áudio vazio")

    pergunta = transcribe(audio_bytes, filename=file.filename or "audio.webm").strip()
    if not pergunta:
        raise HTTPException(status_code=422, detail="não foi possível transcrever o áudio")

    result = _rag_chain(pergunta)
    audio_reply = synthesize(result["resposta"])
    audio_b64 = base64.b64encode(audio_reply).decode("ascii")

    return AskAudioResponse(
        pergunta=pergunta,
        resposta=result["resposta"],
        fontes=result["fontes"],
        audio_base64=audio_b64,
    )
