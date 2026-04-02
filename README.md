# Real-Time AI Meeting Assistant

A real-time AI assistant that joins your meetings, listens to the conversation, and contributes live insights — through chat or spoken replies — while the discussion is still happening.

## What It Does

- Joins Google Meet, Microsoft Teams, or Zoom meetings as a bot participant
- Listens to live transcript in real-time via captions (Meet) or Recall.ai (Teams/Zoom)
- Detects when to contribute: contradictions, memory gaps, confusion, decisions forming, risks being overlooked
- Responds to direct voice commands ("Alex, summarize that") or typed chat commands (`@Alex risks`)
- Speaks replies aloud when addressed by name on Teams/Zoom; uses chat for everything else

## Quick Start

```bash
# 1. Setup
cd meet_ai_agent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
# Fill in OPENAI_API_KEY (required)
# Fill in RECALL_API_KEY (required for Teams/Zoom)

# 3. Run simulation (no meeting needed)
python -m src.main --simulate

# 4. Join a real meeting
python -m src.main -m "https://teams.microsoft.com/l/meetup-join/..."
python -m src.main -m "https://meet.google.com/xxx-yyyy-zzz"
```

### Teams/Zoom Real-Time Setup

Real-time transcript on Teams/Zoom requires a public WebSocket tunnel so Recall.ai can push transcript events:

```bash
# Install cloudflared (one time)
winget install Cloudflare.cloudflared

# Start tunnel before launching the bot
cloudflared tunnel --url http://localhost:8766

# Copy the https://xxx.trycloudflare.com URL
# In .env, set RECALL_REALTIME_URL=wss://xxx.trycloudflare.com
```

Without the tunnel, transcript is only available after the meeting ends.

## How It Works

```
Microphone → Platform STT → Transcript Chunks
                                  ↓
                          Context Engine (sliding window, structured state)
                                  ↓
                          Trigger Engine (rules + local intelligence scoring)
                                  ↓
                          LLM Service (generates reply or SKIP)
                                  ↓
                    ┌─────── Modality Policy ───────┐
                    ↓                               ↓
              Chat Manager                   Speech Service
           (formatted text)              (OpenAI TTS → MP3)
                    ↓                               ↓
              Bot.send_chat()              Bot.send_speech()
                    ↓                          (Recall.ai)
              Meeting Chat                  Meeting Audio
```

**Modality rule:** Only direct voice-addressed replies ("Alex, ...") attempt speech. Everything else stays in chat. If speech fails, it falls back to chat automatically.

## Trigger Types

| Trigger | How It Fires | Response |
|---|---|---|
| Explicit command | `@Alex summarize` in chat | Chat reply |
| Voice address | "Alex, what did Sarah say?" spoken aloud | Speech reply (Teams/Zoom) or chat (Meet) |
| Contradiction | Two speakers state conflicting facts | Proactive chat |
| Memory gap | Someone tries to recall something from earlier | Proactive chat |
| Decision conflict | A decision contradicts a raised concern | Proactive chat |
| Repeated confusion | Multiple people express confusion | Proactive chat |
| Factual inconsistency | Numbers/facts don't match across statements | Proactive chat |
| Topic drift | Conversation wanders from the main subject | Proactive chat |
| Collaboration opportunity | Current discussion relates to an unresolved earlier point | Proactive chat |

## Project Structure

```
src/
├── main.py                    # Orchestrator + main loop
├── config.py                  # All configuration (env-driven)
├── bot/
│   ├── base_bot.py            # Abstract bot interface (chat + speech)
│   ├── google_meet_bot.py     # Playwright browser automation
│   ├── recall_bot.py          # Recall.ai API (Teams/Zoom)
│   └── teams_bot.py           # Teams-specific wrapper
├── captions/
│   └── caption_scraper.py     # Google Meet DOM caption extraction
├── transcription/
│   ├── realtime_server.py     # WebSocket server for Recall transcript push
│   ├── recall_webhook.py      # Webhook receiver for Recall events
│   └── streaming_stt.py       # Mock STT for simulation mode
├── context/
│   ├── models.py              # TranscriptChunk, MeetingContext, Trigger, etc.
│   └── context_engine.py      # Sliding window, structured state extraction
├── triggers/
│   ├── trigger_engine.py      # Rule-based + voice-address detection
│   └── conversation_intelligence.py  # Tier-1 local signal scoring
├── reasoning/
│   ├── llm_service.py         # LLM response generation (OpenAI/Anthropic/Azure)
│   └── speech_service.py      # OpenAI TTS for spoken replies
└── chat/
    └── chat_manager.py        # Delivery routing (chat vs speech), rate limiting
```

## Configuration

All configuration is in `.env`. Key settings:

| Variable | Purpose | Required |
|---|---|---|
| `OPENAI_API_KEY` | LLM responses + TTS for spoken replies | Yes |
| `RECALL_API_KEY` | Teams/Zoom bot via Recall.ai | For Teams/Zoom |
| `RECALL_REALTIME_URL` | WebSocket tunnel for live transcript | For real-time on Teams/Zoom |
| `ENABLE_VOICE_REPLIES` | Enable spoken replies for voice-addressed questions | Default: true |
| `OPENAI_TTS_MODEL` | TTS model (`tts-1` or `tts-1-hd`) | Default: tts-1 |
| `OPENAI_TTS_VOICE` | Voice for spoken replies | Default: alloy |
| `ENABLE_PROACTIVE_MESSAGES` | Allow unsolicited observations | Default: true |
| `MIN_SECONDS_BETWEEN_PROACTIVE` | Cooldown between proactive messages | Default: 30 |

See `.env.example` for the full list.

## Testing

```bash
# Run full test suite (109 tests)
python -m pytest

# Run simulation mode (no API keys needed for trigger/context tests)
python -m src.main --simulate
```

## Architecture Layers

1. **Ingestion** — Captions (Meet) or Recall.ai WebSocket (Teams/Zoom) produce `TranscriptChunk` events
2. **Context** — Sliding transcript window + structured extraction (concerns, decisions, questions, action items)
3. **Triggers** — Rule-based pattern matching + local intelligence scoring, every 500ms
4. **Reasoning** — LLM generates a reply or SKIP; deterministic fallback when LLM is unavailable
5. **Delivery** — Modality policy routes to chat or speech; rate limiting and cooldowns

See [architecture.md](docs/architecture.md) for detailed design.

## License

MIT
