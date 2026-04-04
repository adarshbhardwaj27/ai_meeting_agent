# Real-Time AI Meeting Assistant

A real-time AI assistant that joins your meetings, listens to the conversation, and contributes live insights — through chat or spoken replies — while the discussion is still happening.

## What It Does

- Joins Microsoft Teams or Zoom meetings as a bot participant
- Listens to live transcript in real-time via Recall.ai
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
# Fill in ELEVENLABS_API_KEY (required for spoken replies)
# Fill in RECALL_API_KEY (required for Teams/Zoom)

# 3. Run simulation (no meeting needed)
python -m src.main --simulate

# 4. Join a real meeting
python -m src.main -m "https://teams.microsoft.com/l/meetup-join/..."
python -m src.main -m "https://zoom.us/j/123456789"
```
<!-- python -m src.main -m "https://teams.live.com/meet/9390732530935?p=yXpaWHBxfx1xkgZfXQ" -->

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
                          (formatted text)            (ElevenLabs TTS → MP3)
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
| Voice address | "Alex, what did Sarah say?" spoken aloud | Speech reply (Teams/Zoom) |
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
│   ├── recall_bot.py          # Recall.ai API (Teams/Zoom)
│   └── teams_bot.py           # Teams-specific wrapper
├── transcription/
│   ├── realtime_server.py     # WebSocket server for Recall transcript push
│   └── streaming_stt.py       # Mock STT for simulation mode
├── context/
│   ├── models.py              # TranscriptChunk, MeetingContext, Trigger, etc.
│   └── context_engine.py      # Sliding window, structured state extraction
├── triggers/
│   ├── trigger_engine.py      # Rule-based + voice-address detection
│   └── conversation_intelligence.py  # Tier-1 local signal scoring
├── reasoning/
│   ├── llm_service.py         # LLM response generation (OpenAI/Anthropic/Azure)
│   └── speech_service.py      # ElevenLabs TTS for spoken replies
└── chat/
    └── chat_manager.py        # Delivery routing (chat vs speech), rate limiting
```

## Configuration

All configuration is in `.env`. Key settings:

| Variable | Purpose | Required |
|---|---|---|
| `OPENAI_API_KEY` | LLM responses | Yes |
| `CONTEXT_SUMMARY_MODEL` | Optional separate model for background context cleanup; defaults to `OPENAI_MODEL` | No |
| `ELEVENLABS_API_KEY` | Spoken replies via ElevenLabs TTS | For voice replies |
| `RECALL_API_KEY` | Teams/Zoom bot via Recall.ai | For Teams/Zoom |
| `RECALL_REALTIME_URL` | WebSocket tunnel for live transcript | For real-time on Teams/Zoom |
| `ENABLE_VOICE_REPLIES` | Enable spoken replies for voice-addressed questions | Default: true |
| `ELEVENLABS_MODEL_ID` | TTS model | Default: `eleven_flash_v2_5` |
| `ELEVENLABS_VOICE_ID` | Voice ID for spoken replies | Default: ElevenLabs premade voice |
| `ELEVENLABS_OUTPUT_FORMAT` | ElevenLabs audio format | Default: `mp3_22050_32` for lower speech latency |
| `ELEVENLABS_LATENCY_MODE` | ElevenLabs latency optimization mode | Default: `3` |
| `ENABLE_PROACTIVE_MESSAGES` | Allow unsolicited observations | Default: true |
| `MIN_SECONDS_BETWEEN_PROACTIVE` | Cooldown between proactive messages | Default: 30 |
| `TRIGGER_MODEL_ARTIFACT_PATH` | Optional path to a small offline-trained trigger model artifact | No |
| `ENABLE_TRIGGER_MODEL_LOGGING` | Write fast-model feature vectors and outcomes for offline training | Default: false |
| `TRIGGER_MODEL_LOG_PATH` | JSONL log path for trigger-model training data | Default: `logs/trigger_model_events.jsonl` |

If you want a cheaper model for background memory compaction than for live replies, set `CONTEXT_SUMMARY_MODEL`. If you leave it unset, the assistant reuses `OPENAI_MODEL`.

The trigger engine now includes an artifact-backed fast local scorer for proactive interventions. Keep `TRIGGER_MODEL_ARTIFACT_PATH` empty to use the bundled bootstrap artifact, or point it at a newer offline-trained artifact when available. Enable `ENABLE_TRIGGER_MODEL_LOGGING` when you want to collect JSONL training data from live sessions.

See `.env.example` for the full list.

## Testing

```bash
# Run full test suite (127 tests)
python -m pytest

# Run simulation mode (no API keys needed for trigger/context tests)
python -m src.main --simulate

# Run the browser playground for uploaded JSON scenarios
c:/Users/Adarsh.Bhardwaj/codes/meet_ai_agent/venv/Scripts/python.exe tools/scenario_playground_app.py
# Then open http://127.0.0.1:8769
```

The scenario playground accepts either full E2E-style JSON with an `events` array or simpler uploaded conversation JSON like:

```json
{
        "meeting_id": "quick_demo",
        "conversation": [
                {"speaker": "John", "text": "The deadline is Friday."},
                {"speaker": "Sarah", "text": "The deadline is not Friday, it is Monday.", "delay_ms": 1500}
        ],
        "final_wait_ms": 2200
}
```

It replays the conversation through the real assistant pipeline and shows transcript, trigger, and response events in the browser without requiring you to save the case as a permanent local E2E file.

## Architecture Layers

1. **Ingestion** — Captions (Meet) or Recall.ai WebSocket (Teams/Zoom) produce `TranscriptChunk` events
2. **Context** — Sliding transcript window + structured extraction (concerns, decisions, questions, action items)
3. **Triggers** — Rule-based pattern matching + local intelligence scoring, every 500ms
4. **Reasoning** — LLM generates a reply or SKIP; deterministic fallback when LLM is unavailable
5. **Delivery** — Modality policy routes to chat or speech; rate limiting and cooldowns

See [architecture.md](docs/architecture.md) for detailed design.

## License

MIT
