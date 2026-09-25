"""VADER sentiment scoring over JSON.

Hardened 2026-09-25:
  - Auth required on /analyze via the X-API-Key header (VADER_API_KEY config var).
  - Lexicon loaded from the vendored copy in nltk_data/, never downloaded at import.
  - VADER's mutable class-level constants copied onto the instance, so tuning one
    analyzer can never leak into another in the same process.
"""

import copy
import hmac
import os
from pathlib import Path
from typing import List, Optional

import nltk
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel

# Load the lexicon from the copy committed to this repo. NLTK searches data.path in
# order, so putting the vendored directory first means SentimentIntensityAnalyzer()
# resolves locally and makes no network call. The previous version called
# nltk.download() at import time, which reached out to NLTK's servers on every dyno
# boot and would fail to start if they were unreachable.
_VENDORED_NLTK_DATA = Path(__file__).resolve().parent / "nltk_data"
if str(_VENDORED_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(_VENDORED_NLTK_DATA))

from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402

# VaderConstants keeps these as CLASS attributes, so mutating them through any
# instance mutates them for every instance in the process. Anything that tunes the
# lexicon or the negator list later (domain vocabulary, threshold work) must not be
# able to silently change the behavior of a scorer it does not own.
_MUTABLE_CONSTANTS = ("NEGATE", "BOOSTER_DICT", "PUNC_LIST", "SPECIAL_CASE_IDIOMS")

# Reading the key at import is deliberate: a missing key should be a visible 503 on
# every request rather than an endpoint that quietly serves unauthenticated traffic.
API_KEY = os.environ.get("VADER_API_KEY", "")


def build_analyzer() -> SentimentIntensityAnalyzer:
    """Return an analyzer whose mutable constants are its own, not the class's."""
    analyzer = SentimentIntensityAnalyzer()
    for name in _MUTABLE_CONSTANTS:
        setattr(
            analyzer.constants,
            name,
            copy.deepcopy(getattr(analyzer.constants, name)),
        )
    return analyzer


app = FastAPI(title="VADER Sentiment API")
sia = build_analyzer()


class Item(BaseModel):
    id: Optional[str] = None
    text: str


class AnalyzeRequest(BaseModel):
    items: List[Item]


def require_api_key(x_api_key: Optional[str]) -> None:
    """Fail closed. An unset key is a misconfiguration, not permission to serve."""
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not configured: VADER_API_KEY is unset.",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
        )


@app.post("/analyze")
def analyze(
    req: AnalyzeRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    results = []
    for item in req.items:
        score = sia.polarity_scores(item.text)
        sentiment = (
            "Positive"
            if score["compound"] > 0.05
            else "Negative"
            if score["compound"] < -0.05
            else "Neutral"
        )
        results.append(
            {
                "id": item.id,
                "text": item.text,
                "compound": score["compound"],
                "pos": score["pos"],
                "neu": score["neu"],
                "neg": score["neg"],
                "sentiment": sentiment,
            }
        )
    return {"results": results}


@app.get("/")
def health():
    """Unauthenticated so uptime checks work. Reports config state, never the key."""
    return {
        "status": "ok",
        "service": "VADER Sentiment API",
        "auth_configured": bool(API_KEY),
        "lexicon_entries": len(sia.lexicon),
    }
