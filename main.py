import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List
from datetime import datetime, timezone
import requests

from database import create_document, get_documents, db

app = FastAPI(title="Veritas - AI Fake News & Plagiarism API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Text to analyze")
    title: Optional[str] = Field(None, description="Optional title or headline")
    source_url: Optional[str] = Field(None, description="Optional source URL")


class AnalyzeResponse(BaseModel):
    type: Literal["fake_news", "plagiarism"]
    score: float
    verdict: str
    details: Dict[str, Any]
    id: Optional[str] = None
    created_at: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "Veritas API running", "endpoints": ["/api/analyze/fake-news", "/api/analyze/plagiarism", "/api/analyses"]}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ------------- AI Helper (OpenAI-compatible optional) -------------

def call_llm(classifier: Literal["fake_news", "plagiarism"], text: str, title: Optional[str] = None, source_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Call an OpenAI-compatible chat/completions endpoint if env vars exist.
    Returns a dict with keys: score (0-1), verdict (str), details (dict).
    If not configured or error occurs, returns None.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return None

    system = (
        "You are Veritas, an academic integrity assistant. "
        "Return a strict JSON object with keys: score (0..1), verdict, details. "
        "Score near 1.0 means highly likely the label applies."
    )

    if classifier == "fake_news":
        user = (
            f"Classify if the following claim or article is likely misinformation or fabricated.\n"
            f"Title: {title or 'N/A'}\nSource: {source_url or 'N/A'}\n\nText:\n{text}\n"
            "Explain reasoning briefly in details.reason, list key indicators in details.indicators (array)."
        )
    else:
        user = (
            "Estimate plagiarism likelihood of the given text (without web access). "
            "Consider repetitiveness, cliched phrases, citation patterns, and writing artifacts.\n"
            f"Title: {title or 'N/A'}\nSource: {source_url or 'N/A'}\n\nText:\n{text}\n"
            "Return details with a short rationale and suspected patterns."
        )

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            },
            timeout=20,
        )
        if resp.status_code >= 400:
            return None
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = None
        try:
            parsed = requests.utils.json.loads(content)  # type: ignore
        except Exception:
            # fallback if SDK changed
            import json as _json
            try:
                parsed = _json.loads(content)
            except Exception:
                return None
        if not isinstance(parsed, dict):
            return None
        # Normalize
        score = float(parsed.get("score", 0))
        verdict = str(parsed.get("verdict", "Unknown"))
        details = parsed.get("details", {})
        return {"score": max(0, min(score, 1)), "verdict": verdict, "details": details}
    except Exception:
        return None


def heuristic_fake_news(text: str) -> Dict[str, Any]:
    flags = []
    score = 0.2
    lowered = text.lower()
    sensational = ["shocking", "you won't believe", "miracle", "cure", "exposed", "secret"]
    for w in sensational:
        if w in lowered:
            flags.append(f"Sensational phrase: {w}")
            score += 0.1
    if any(k in lowered for k in ["no sources", "trust me", "forward this to everyone"]):
        flags.append("Lack of sourcing cues")
        score += 0.1
    if len(text) < 280:
        flags.append("Short, tweet-like claim")
        score += 0.05
    score = max(0.0, min(score, 0.95))
    verdict = "Likely misinformation" if score > 0.6 else ("Unclear" if score > 0.4 else "Likely factual/benign")
    return {"score": score, "verdict": verdict, "details": {"indicators": flags, "reason": "Heuristic estimate (no external model)"}}


def heuristic_plagiarism(text: str) -> Dict[str, Any]:
    # Naive repetition and complexity check
    words = [w.strip('.,;:!?()"\'\n').lower() for w in text.split()]
    unique = len(set(words)) + 1
    repetitiveness = 1 - min(1.0, unique / (len(words) + 1))
    avg_len = sum(len(w) for w in words) / (len(words) + 1)
    score = 0.2 + 0.4 * repetitiveness + (0.1 if avg_len < 4.2 else 0)
    verdict = "Possible plagiarism" if score > 0.55 else ("Borderline" if score > 0.45 else "Unlikely plagiarism")
    return {"score": round(min(score, 0.95), 3), "verdict": verdict, "details": {"repetitiveness": round(repetitiveness, 3), "avg_token_len": round(avg_len, 2), "reason": "Heuristic estimate (no external model)"}}


# ------------- Routes -------------

@app.post("/api/analyze/fake-news", response_model=AnalyzeResponse)
def analyze_fake_news(payload: AnalyzeRequest):
    if not payload.text or len(payload.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Please provide at least 20 characters of text.")

    ai = call_llm("fake_news", payload.text, payload.title, payload.source_url)
    result = ai or heuristic_fake_news(payload.text)

    # Persist
    doc = {
        "type": "fake_news",
        "title": payload.title,
        "source_url": payload.source_url,
        "text": payload.text,
        "score": float(result["score"]),
        "verdict": result["verdict"],
        "details": result.get("details", {}),
    }
    try:
        inserted_id = create_document("analysis", doc)
    except Exception:
        inserted_id = None

    return AnalyzeResponse(
        type="fake_news",
        score=float(result["score"]),
        verdict=result["verdict"],
        details=result.get("details", {}),
        id=inserted_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/analyze/plagiarism", response_model=AnalyzeResponse)
def analyze_plagiarism(payload: AnalyzeRequest):
    if not payload.text or len(payload.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Please provide at least 20 characters of text.")

    ai = call_llm("plagiarism", payload.text, payload.title, payload.source_url)
    result = ai or heuristic_plagiarism(payload.text)

    doc = {
        "type": "plagiarism",
        "title": payload.title,
        "source_url": payload.source_url,
        "text": payload.text,
        "score": float(result["score"]),
        "verdict": result["verdict"],
        "details": result.get("details", {}),
    }
    try:
        inserted_id = create_document("analysis", doc)
    except Exception:
        inserted_id = None

    return AnalyzeResponse(
        type="plagiarism",
        score=float(result["score"]),
        verdict=result["verdict"],
        details=result.get("details", {}),
        id=inserted_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/analyses")
def list_analyses(type: Optional[str] = None, limit: int = 20):
    filt = {"type": type} if type in ("fake_news", "plagiarism") else {}
    try:
        docs = get_documents("analysis", filt, min(max(limit, 1), 50))
    except Exception:
        docs = []
    # Normalize ObjectId and datetimes for frontend
    items: List[Dict[str, Any]] = []
    for d in docs:
        d = dict(d)
        d["id"] = str(d.pop("_id", ""))
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        if isinstance(d.get("updated_at"), datetime):
            d["updated_at"] = d["updated_at"].isoformat()
        items.append(d)
    return {"items": items}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
