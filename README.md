# Real-Time Speech Spam Detection System

A Flask-based application that transcribes live audio and detects phone scams in real time using a multi-layer AI pipeline.

---

## How It Works

Detection runs through four layers in order, falling back if a layer is unavailable:

1. **GPT-4o** — Phone-call-aware LLM classification (requires `OPENAI_API_KEY`)
2. **Semantic Embeddings** — Cosine similarity against 24 scam + 14 ham reference phrases (requires `OPENAI_API_KEY`)
3. **Hotword Detection** — Severity-weighted keyword matching with negation awareness (e.g. "your account is secure" won't trigger)
4. **Rule-Based** — Regex patterns for urgency, gift cards, threats, and social engineering

Transcription uses **OpenAI Whisper** locally (no internet needed) with optional **Deepgram** for faster real-time monitoring (~300ms vs ~5s).

Multi-language support: English, Spanish, Hindi, French, Arabic, Portuguese — Whisper auto-detects the language.

---

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Production-ready web app |
| `extra` | Phase 3 extras: browser extension, REST API, multi-language |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Whisper model

```bash
python scripts/download_model.py
```

This downloads the `base` model (~140MB) to `~/.cache/whisper/`. Required for offline transcription.

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your keys
```

The app works fully offline without any API keys — detection falls back to hotword + rule-based methods.

### 4. Run

```bash
python main.py
```

Open `http://localhost:5000`.

---

## Features

- **Upload or record** audio directly in the browser
- **Real-time monitoring** mode — streams microphone in 5-second chunks with rolling context
- **Demo samples** — one-click IRS scam, bank fraud, lottery, and safe call scripts
- **Phrase highlighting** — matched hotwords color-coded by severity
- **Confidence gauge** — SVG arc gauge with green/yellow/red zones
- **Session history** — all analyses logged within the session
- **Language badge** — shown when a non-English language is detected

---

## Optional Integrations

### Deepgram (faster transcription)

Set `DEEPGRAM_API_KEY` in `.env`. Cuts monitoring latency from ~5s to ~300ms.

### Plivo (phone call integration)

Set `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_FROM_NUMBER`, and `PLIVO_ALERT_NUMBER` in `.env`.

Routes activated: `POST /plivo/answer`, `POST /plivo/recording`, `GET /plivo/status`

When a scam is detected on an incoming call, an SMS alert is sent to `PLIVO_ALERT_NUMBER`.

### REST API

Always enabled at `/api/v1/`. Optionally protected with `API_KEYS` env var (comma-separated).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Status and capabilities |
| `/api/v1/analyze` | POST | Upload audio file for analysis |
| `/api/v1/analyze-text` | POST | Analyze text directly |
| `/api/v1/docs` | GET | Swagger UI |
| `/api/v1/openapi.json` | GET | OpenAPI 3.0 spec |

### Browser Extension (`extension/`)

Chrome extension (Manifest V3) that monitors audio on Google Meet, Zoom, and Microsoft Teams. Captures 5-second chunks, sends to the backend, and shows a floating overlay with live verdicts.

To install: open `chrome://extensions`, enable Developer Mode, click **Load unpacked**, select the `extension/` folder.

---

## Project Structure

```
├── app.py                  # Flask app, routes, detection pipeline
├── main.py                 # Entry point, preloads Whisper models
├── requirements.txt
├── .env.example
├── utils/
│   ├── transcriber.py      # Whisper wrapper
│   ├── hotword_detector.py # Negation-aware hotword matching
│   ├── hotwords_data.py    # English hotword severity scores
│   ├── multilang_hotwords.py # ES/HI/FR/AR/PT hotwords
│   ├── embedding_detector.py # Semantic similarity detector
│   ├── rule_based_detector.py # Regex fallback
│   ├── api_handler.py      # REST API blueprint
│   └── plivo_handler.py    # Plivo phone call blueprint
├── extension/              # Chrome extension (Phase 3A)
├── scripts/
│   └── download_model.py   # Whisper model downloader
├── static/
│   ├── css/styles.css
│   └── js/main.js
└── templates/
    └── index.html
```

---

## Environment Variables

See `.env.example` for all variables with descriptions.

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | No | GPT-4o + embeddings detection layers |
| `DEEPGRAM_API_KEY` | No | Fast transcription for monitoring |
| `PLIVO_AUTH_ID` | No | Enables phone call integration |
| `PLIVO_AUTH_TOKEN` | No | Plivo auth |
| `PLIVO_FROM_NUMBER` | No | Plivo caller number |
| `PLIVO_ALERT_NUMBER` | No | Number to SMS on scam detection |
| `SESSION_SECRET` | Recommended | Flask session signing key |
| `API_KEYS` | No | Comma-separated keys to protect REST API |
