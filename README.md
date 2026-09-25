# vader-sentiment-api

Internal sentiment-scoring service operated by MDLMN Technologies LLC. A Heroku-hosted FastAPI wrapper around NLTK's VADER, exposing a JSON API for batch scoring of short text.

Access is restricted by API key. This is not a public endpoint.

**Status:** scoped as the first-pass ("tier 1") scorer for MDLMN's sentiment-triggered service recovery work (DEV-594). No production consumers call it yet. See Scope and limitations below before relying on its output.

---

## Authentication

`POST /analyze` requires an `X-API-Key` header matching the `VADER_API_KEY` config var.

The service **fails closed**: if `VADER_API_KEY` is unset, `/analyze` returns `503` rather than serving unauthenticated traffic. `GET /` stays open so uptime checks work, and reports whether auth is configured without ever revealing the key.

```bash
heroku config:set VADER_API_KEY='<value>' -a vader-sentiment-api
```

## Usage

```bash
curl -X POST https://vader-sentiment-api-779f011b0282.herokuapp.com/analyze \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $VADER_API_KEY" \
  -d '{"items":[{"id":"1","text":"You guys are amazing, thank you!"}]}'
```

Response:

```json
{"results":[{"id":"1","text":"You guys are amazing, thank you!",
  "compound":0.7644,"pos":0.583,"neu":0.417,"neg":0.0,"sentiment":"Positive"}]}
```

Health check:

```bash
curl https://vader-sentiment-api-779f011b0282.herokuapp.com/
# {"status":"ok","service":"VADER Sentiment API","auth_configured":true,"lexicon_entries":7502}
```

## Notes on the lexicon

The VADER lexicon is **committed to this repo** at `nltk_data/sentiment/vader_lexicon.zip` and loaded from there. The service makes no network call to fetch it, so a dyno restart cannot fail because NLTK's servers are unreachable.

Dependencies are pinned in `requirements.txt`. Keep them pinned: VADER's `compound` scores feed downstream thresholds, and an unpinned rebuild can shift scoring underneath them.

## A note on tuning

`VaderConstants` keeps `NEGATE`, `BOOSTER_DICT`, `PUNC_LIST` and `SPECIAL_CASE_IDIOMS` as **class** attributes, so mutating them through one analyzer instance changes every instance in the process. `build_analyzer()` copies them onto the instance to prevent that. If you add lexicon or negator tuning, build analyzers through `build_analyzer()` and never mutate `VaderConstants` directly.

Also note that VADER negation multiplies a valence by `-0.74` and **flips the sign in either direction**, so adding a negator can turn a word you intended to be negative into a positive contribution. Tune against a fixed labeled corpus with a regression check, not by intuition.

---

## 📜 Data Handling
Text submitted for scoring is processed in memory and returned in the response. The service writes nothing to storage: there is no database, no file output, and no analytics. Request and response bodies are not logged.

Standard HTTP access logging (method, path, status code, timing) is retained by the platform. That logging does not include message content.

All requests are transmitted over HTTPS. Because submitted text may include customer conversation content, callers are responsible for ensuring their own retention and consent obligations are met before sending it.

## 📜 Scope and Limitations
This service is for internal MDLMN use. It is not offered to third parties, and the API key is not distributed outside MDLMN.

Scoring comes from the open-source NLTK VADER model, which is lexicon-based: it scores affect vocabulary and has no understanding of context. Measured limitations, from testing against MDLMN's own verticals on 2026-09-25:

- **Stock VADER did not detect a single one of seven genuinely negative customer messages** at any strong-negative threshold tested. Complaints that narrate a factual failure without affect words ("Nobody showed up", "You forgot half my order again") score at or near `0.000`.
- **Sarcasm scores confidently positive.** "Great. Another text reminder. Just what I needed today." returns `+0.625`. Lexicon tuning does not fix this, because the inversion is contextual rather than lexical.
- **Domain vocabulary is absent.** `redeem`, `renew`, `forfeit`, `pawn`, `loan`, `ticket` and `due` are not in the lexicon at all, and `interest` carries `+2.0`, which misreads pawn finance language as enthusiasm.

Treat output as a cheap first-pass signal, not a verdict. A confident reading is not the same as a correct one, so do not route on the score alone. Any consumer needs an escalation path for gray-zone and confidently-wrong cases.

Results are not guaranteed accurate. Callers are responsible for downstream use of the outputs.
