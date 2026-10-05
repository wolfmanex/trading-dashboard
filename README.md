---
title: AI Trading Dashboard
emoji: 📈
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# AI Trading Dashboard

Streamlit dashboard with multi-timeframe charts, a small-cap breakout scanner, an AI trade reviewer
(Gemini) with follow-up questions, a market conditions panel, a trade journal and a forward track record
of the scanner's signals.

GitHub Actions run the weekday pre-market scan (Telegram top 10 plus an AI brief), then a market session
with breakout and journal stop/target alerts every 15 minutes, then an evening recap. GitHub's own cron never
fired in this repo, so a daily Claude routine starts the scan (weekdays 8:33 New York) and each step starts
the next; the scan's cron stays as a backup and a second scan on the same day stands down.

The block at the top of this file is the Hugging Face Spaces configuration: the Space builds the
`Dockerfile` and serves the app on port 7860.

## Running it

- Locally: `pip install -r requirements.txt`, put your keys in `.streamlit/secrets.toml`, then
  `streamlit run app.py`.
- Docker: `docker build -t trading-dashboard .` then
  `docker run -p 7860:7860 -e GEMINI_API_KEY=... trading-dashboard`.
- Hugging Face Spaces: the `Sync to Hugging Face Space` workflow pushes `main` to the Space on every merge.
  Keys go in the Space's secrets (Settings → Variables and secrets); the app reads them as environment variables.

## Keys

| Name | Used for |
|---|---|
| `GEMINI_API_KEY` | AI trade review and scanner grading |
| `GEMINI_MODEL` | Optional: pin a Gemini model (default `gemini-flash-latest`) |
| `FINNHUB_API_KEY` | Optional: earnings dates |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Scheduled scan, alerts, watchdog and recap (GitHub Actions secrets) |
| `GITHUB_JOURNAL_TOKEN` | App only: fine-grained GitHub token with read/write access to this repo's contents, so the trade journal is saved on the `signal-log` branch where the alerts read it. Without it the journal is a local `journal.csv`. |
