# Pydantic request/response models for the HTTP API.


from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Incoming question from a client."""

    question: str = Field(..., min_length=1)


class SourceInfo(BaseModel):
    """A document/section that contributed to an answer."""

    source: str
    section: str


class AnswerResponse(BaseModel):
    """A synthesized answer plus its supporting sources."""

    answer: str
    sources: list[SourceInfo]


class HealthResponse(BaseModel):
    """Service liveness and index status."""

    status: str
    documents_indexed: int
