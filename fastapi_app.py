from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon once on startup
nltk.download("vader_lexicon")

app = FastAPI(title="VADER Sentiment API")

class Item(BaseModel):
    id: Optional[str] = None
    text: str

class AnalyzeRequest(BaseModel):
    items: List[Item]

sia = SentimentIntensityAnalyzer()

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    results = []
    for item in req.items:
        score = sia.polarity_scores(item.text)
        sentiment = (
            "Positive" if score["compound"] > 0.05
            else "Negative" if score["compound"] < -0.05
            else "Neutral"
        )
        results.append({
            "id": item.id,
            "text": item.text,
            "compound": score["compound"],
            "pos": score["pos"],
            "neu": score["neu"],
            "neg": score["neg"],
            "sentiment": sentiment
        })
    return {"results": results}
