# FastAPI application exposing the CogniCore Q&A endpoints.

# Run with::

#     uvicorn app.main:app --reload



from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.rag import ask
from app.schemas import AnswerResponse, HealthResponse, QuestionRequest
from app.vectorstore import vector_store

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="CogniCore Q&A API")

# Open CORS for local development / the bundled frontend.
# NOTE: restrict allow_origins to known hosts before deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest) -> AnswerResponse:
    """Answer a question from the indexed document corpus."""
    try:
        result = ask(request.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AnswerResponse(**result)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service status and how many chunks are indexed."""
    return HealthResponse(status="ok", documents_indexed=vector_store.count())


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")
