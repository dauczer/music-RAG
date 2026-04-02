from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.chain import ask, compare_artists, route_and_ask
from rag.vectorstore import index_artist

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    artist: str


class AskRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    artist: str
    question: str


class CompareRequest(BaseModel):
    artist1: str
    artist2: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def index(req: IndexRequest):
    try:
        newly_indexed = index_artist(req.artist)
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=f"No scraped data found for {req.artist}. Run the scraper first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if newly_indexed:
        return {"message": f"{req.artist} indexed successfully."}
    return {"message": f"{req.artist} was already indexed."}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    try:
        answer = route_and_ask(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"answer": answer}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        answer = ask(req.artist, req.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"answer": answer}


@app.post("/compare")
def compare(req: CompareRequest):
    try:
        answer = compare_artists(req.artist1, req.artist2, req.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"answer": answer}
