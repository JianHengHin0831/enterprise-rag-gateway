"""Shared data models used across modules and the HTTP API."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    INTERN = "intern"
    EMPLOYEE = "employee"
    ENG = "eng"
    HR = "hr"
    SALES = "sales"
    FINANCE = "finance"


# Documents visible to every role in the tenant carry this value in `allowed_roles`.
PUBLIC_ACCESS = "all"


class Identity(BaseModel):
    """Caller identity. Construct ONLY from a verified JWT (app/security/auth.py)."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    role: Role
    user_id: str


class DevTokenRequest(BaseModel):
    """Dev-only persona switch for the demo UI (simulates the upstream SSO)."""

    tenant_id: str = Field(min_length=1, max_length=64)
    role: Role
    user_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")


class DevTokenResponse(BaseModel):
    token: str
    expires_in: int


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    stream: bool = False


class Citation(BaseModel):
    index: int
    doc_id: str
    chunk_index: int
    title: str | None = None


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    cache_hit: bool = False
    model_used: str | None = None
    degraded: bool = False
    citation_mismatch: bool = False
    # USD for this request's answer call; 0 for cache hits and refusals without an LLM call,
    # None while the model's price is not configured in configs/models.yaml.
    cost_usd: float | None = None


class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_index: int
    text: str
    title: str | None = None
    score: float = 0.0
    rerank_score: float | None = None
    suspicious: bool = False

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{self.chunk_index}"
