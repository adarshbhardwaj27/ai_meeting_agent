# Real-Time AI Meeting Assistant

## 🎯 What This Is

A real-time thinking assistant that quietly nudges your team in chat **while the conversation is still happening**.

**This is NOT:** Meeting notes + AI summary after the meeting  
**This IS:** Live, contextual insights injected into meeting chat in real-time

## 🚀 Quick Start

```bash
# 1. Clone and setup
cd meet_ai_agent
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
# Edit .env with your API keys

# 3. Run
python -m src.main --meeting-url "https://meet.google.com/xxx-xxxx-xxx"
```

## 🧠 Core Behaviors

### Proactive Messages (Real-Time)
The AI automatically sends chat messages when:
- Discussion contradicts earlier statements
- A decision is forming but risks were mentioned
- The team appears headed wrong direction
- Complex concepts need clarification
- Same confusion appears multiple times

### Reactive Messages (On Command)
Respond to explicit commands:
- `@Alex summarize what we discussed`
- `@Alex what are the risks here`
- `@Alex explain this term`
- `@Alex what decision are we leaning toward`

## 💬 Message Style

All AI messages are:
- **1-3 lines max**
- **Neutral and factual**
- **Suggestive, not commanding**
- **Easy to ignore**
- **Clearly labeled:** `Alex (AI):`

## 📁 Project Structure

```
meet_ai_agent/
├── docs/
│   └── architecture.md          # System design
├── src/
│   ├── main.py                  # Entry point & orchestrator
│   ├── config.py                # Configuration
│   ├── bot/                     # Meeting platform bots
│   │   ├── base_bot.py          # Abstract interface
│   │   ├── google_meet_bot.py   # Google Meet
│   │   ├── teams_bot.py         # MS Teams
│   │   └── zoom_bot.py          # Zoom
│   ├── transcription/
│   │   └── streaming_stt.py     # Real-time speech-to-text
│   ├── context/
│   │   ├── models.py            # Data structures
│   │   └── context_engine.py    # Live context management
│   ├── triggers/
│   │   └── trigger_engine.py    # Real-time trigger detection
│   ├── reasoning/
│   │   └── llm_service.py       # LLM integration
│   └── chat/
│       └── chat_manager.py      # Chat message handling
├── tests/
│   └── test_simulation.py       # End-to-end simulation
├── requirements.txt
└── .env.example
```

## 🔑 Configuration

See `.env.example` for all configuration options:
- **LLM Provider:** OpenAI, Anthropic, or Azure OpenAI
- **STT Provider:** Deepgram (recommended), Google, or Azure
- **Meeting Platforms:** Google Meet, Teams, Zoom credentials
- **Behavior Tuning:** Context window, trigger sensitivity, message frequency

## 📖 Documentation

- [System Architecture](docs/architecture.md) - Detailed design and data flow
- [API Reference](docs/api.md) - Component APIs (TODO)
- [Deployment Guide](docs/deployment.md) - Production setup (TODO)

## ✅ MVP Success Criteria

- [ ] AI posts useful chat messages DURING meetings
- [ ] Messages appear at the right moments
- [ ] Users do not feel interrupted
- [ ] The AI feels like a quiet, helpful teammate

## 📜 License

MIT
