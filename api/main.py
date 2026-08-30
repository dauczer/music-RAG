import logging
import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag.chain import RagResult, ask_result, compare_artists, route_and_ask
from rag.vectorstore import list_artist_summaries

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

_trust_proxy_headers = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"
_demo_rate_limit = os.getenv("DEMO_RATE_LIMIT", "3/minute")


def _rate_limit_key(request: Request) -> str:
    if _trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        client_ip = forwarded_for.split(",", maxsplit=1)[0].strip()
        if client_ip:
            return client_ip
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class ChatRequest(BaseModel):
    artist: str = Field(..., min_length=1, max_length=500)
    question: str = Field(..., min_length=1, max_length=500)


class CompareRequest(BaseModel):
    artist1: str = Field(..., min_length=1, max_length=500)
    artist2: str = Field(..., min_length=1, max_length=500)
    question: str = Field(..., min_length=1, max_length=500)


class ChatResponse(RagResult):
    request_id: str


class ArtistResponse(BaseModel):
    slug: str
    name: str
    song_count: int


class ArtistsResponse(BaseModel):
    artists: list[ArtistResponse]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/artists", response_model=ArtistsResponse)
def artists() -> ArtistsResponse:
    return ArtistsResponse(
        artists=[
            ArtistResponse(slug=artist.slug, name=artist.name, song_count=artist.song_count)
            for artist in list_artist_summaries()
        ]
    )


@app.post("/ask")
@limiter.limit(_demo_rate_limit)
def ask_endpoint(request: Request, req: AskRequest):
    try:
        answer = route_and_ask(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"answer": answer}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(_demo_rate_limit)
def chat(request: Request, req: ChatRequest) -> ChatResponse:
    try:
        result = ask_result(req.artist, req.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return ChatResponse(**result.model_dump(), request_id=str(uuid4()))


@app.post("/compare")
@limiter.limit(_demo_rate_limit)
def compare(request: Request, req: CompareRequest):
    try:
        answer = compare_artists(req.artist1, req.artist2, req.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"answer": answer}
