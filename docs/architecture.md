# Real-Time AI Meeting Assistant - System Architecture

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEETING PLATFORM                                   │
│                    (Google Meet / Teams / Zoom)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────┐                              ┌──────────────┐          │
│    │  Audio Feed  │                              │  Chat Feed   │          │
│    └──────┬───────┘                              └──────┬───────┘          │
│           │                                             │                   │
└───────────┼─────────────────────────────────────────────┼───────────────────┘
            │                                             │
            ▼                                             ▼
┌───────────────────────┐                    ┌───────────────────────┐
│   MEETING BOT         │                    │   CHAT READER         │
│   - Join as bot       │                    │   - Monitor messages  │
│   - Capture audio     │                    │   - Detect @mentions  │
│   - Send chat         │◄───────────────────│   - Parse commands    │
└───────────┬───────────┘                    └───────────┬───────────┘
            │                                             │
            ▼                                             │
┌───────────────────────┐                                │
│   STREAMING STT       │                                │
│   - Deepgram/Google   │                                │
│   - Real-time chunks  │                                │
│   - Speaker labels    │                                │
└───────────┬───────────┘                                │
            │                                             │
            ▼                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVENT BUS (Async)                                  │
│   Events: TranscriptChunk, ChatMessage, TriggerFired, ResponseReady         │
└───────────┬───────────────────────────────────────────────────────────────┬─┘
            │                                                               │
            ▼                                                               ▼
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│   CONTEXT ENGINE      │    │   TRIGGER ENGINE      │    │   CHAT MANAGER        │
│                       │    │                       │    │                       │
│   - Sliding window    │◄──►│   - Decision detect   │───►│   - Format messages   │
│   - Compressed summary│    │   - Conflict detect   │    │   - Rate limiting     │
│   - State tracking    │    │   - Drift detect      │    │   - Send to platform  │
│   - Topic extraction  │    │   - Command parse     │    │                       │
└───────────┬───────────┘    └───────────┬───────────┘    └───────────────────────┘
            │                            │
            └──────────┬─────────────────┘
                       ▼
            ┌───────────────────────┐
            │   LLM REASONING       │
            │                       │
            │   - Context + trigger │
            │   - Generate response │
            │   - 1-3 lines max     │
            └───────────────────────┘
```

---

## 📊 Event-Driven Data Flow

### Timeline of a Single Conversation Turn

```
TIME        EVENT                          COMPONENT ACTION
─────────────────────────────────────────────────────────────────────────────
0ms         Audio captured                 MeetingBot → AudioBuffer
50ms        Audio chunk sent               AudioBuffer → StreamingSTT
100ms       Transcript chunk received      StreamingSTT → EventBus
            "We should probably use..."    
            
120ms       Context updated                ContextEngine receives chunk
            - Added to sliding window
            - Topic detected: "architecture"
            
150ms       Trigger check                  TriggerEngine scans for patterns
            - Decision language: "should"
            - No trigger yet (incomplete)
            
200ms       More transcript                "...Kafka for this"
            
220ms       Context updated                ContextEngine
            - Full sentence parsed
            - Decision candidate detected
            
250ms       Trigger fires!                 TriggerEngine
            - Type: DECISION_FORMING
            - Reason: "Kafka decision forming"
            - Related concerns: ["latency issues"]
            
300ms       LLM request                    LLMService
            - Input: recent transcript + context + trigger
            - Streaming response started
            
450ms       Response ready                 LLMService → ChatManager
            "Earlier latency concerns may conflict with Kafka"
            
500ms       Chat sent                      ChatManager → MeetingBot
            Alex (AI): Earlier latency concerns may conflict...
```

---

## 🧱 Component Details

### 1. Meeting Bot (`src/bot/`)

**Purpose:** Interface with meeting platforms

**Responsibilities:**
- Join meeting as a silent participant
- Capture live audio stream
- Read chat messages in real-time
- Send chat messages to meeting

**Platform-Specific Implementations:**

| Platform | Audio Capture | Chat API | Notes |
|----------|--------------|----------|-------|
| Google Meet | Web Audio API via Puppeteer | Meet Chat API | Requires bot account |
| MS Teams | Bot Framework SDK | Graph API | Requires Azure Bot registration |
| Zoom | Meeting SDK | Zoom Chat API | Requires Zoom App |

```python
# Abstract interface
class MeetingBot(ABC):
    async def join(self, meeting_url: str) -> None
    async def leave(self) -> None
    async def send_chat(self, message: str) -> None
    def on_audio_chunk(self, callback: Callable) -> None
    def on_chat_message(self, callback: Callable) -> None
```

### 2. Streaming Speech-to-Text (`src/transcription/`)

**Purpose:** Convert audio to text in real-time

**Requirements:**
- Latency < 500ms
- Streaming (not batch)
- Speaker diarization (if available)
- Interim results for faster feedback

**Recommended: Deepgram**
- WebSocket-based streaming
- ~200ms latency
- Speaker labels
- Punctuation & formatting

```python
# Data structure for transcript chunks
@dataclass
class TranscriptChunk:
    text: str
    speaker: Optional[str]
    timestamp: datetime
    is_final: bool
    confidence: float
```

### 3. Context Engine (`src/context/`)

**Purpose:** Maintain live understanding of the conversation

**Data Structures:**

```python
@dataclass
class MeetingContext:
    # Raw transcript - sliding window (last ~10 min)
    recent_transcript: List[TranscriptChunk]
    
    # Compressed summary of earlier discussion
    historical_summary: str
    
    # Tracked state
    current_topic: str
    topics_discussed: List[str]
    concerns_raised: List[Concern]
    decisions_made: List[Decision]
    decisions_pending: List[Decision]
    open_questions: List[str]
    
    # Metadata
    participants: Set[str]
    meeting_duration: timedelta
    last_updated: datetime
```

**Update Strategy:**
- Transcript chunks: Added immediately
- State tracking: Updated every 5 seconds
- Historical summary: Compressed when window exceeds threshold

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CONTEXT TIMELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [=== COMPRESSED SUMMARY ===][======= SLIDING WINDOW =======]      │
│  (older than 10 min)         (last 10 minutes - full text)         │
│                                                                     │
│  "The team discussed         "John: So for the database, I think   │
│   initial architecture        we should consider PostgreSQL.        │
│   options, raised concerns    Sarah: What about the scaling issue?  │
│   about latency..."           John: Good point, maybe we need..."   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. Trigger Engine (`src/triggers/`)

**Purpose:** Detect when AI should send a message

**Trigger Types:**

| Type | Pattern Examples | Priority |
|------|------------------|----------|
| `EXPLICIT_COMMAND` | "@Alex summarize", "@Alex risks" | HIGHEST |
| `DECISION_FORMING` | "let's go with", "we'll do", "I vote for" | HIGH |
| `CONFLICT_DETECTED` | Statement contradicts prior concern | HIGH |
| `TOPIC_DRIFT` | Discussion moved away from agenda | MEDIUM |
| `REPEATED_CONFUSION` | Same question asked 2+ times | MEDIUM |
| `COMPLEX_CONCEPT` | Technical jargon detected | LOW |

**Detection Logic:**

```python
class TriggerEngine:
    async def check_triggers(self, context: MeetingContext) -> List[Trigger]:
        triggers = []
        
        # 1. Check for explicit commands (highest priority)
        if cmd := self._detect_command(context.recent_transcript):
            triggers.append(Trigger(type=TriggerType.EXPLICIT_COMMAND, ...))
        
        # 2. Check for decision language
        if decision := self._detect_decision(context.recent_transcript):
            # Cross-reference with prior concerns
            conflicts = self._find_conflicts(decision, context.concerns_raised)
            if conflicts:
                triggers.append(Trigger(type=TriggerType.CONFLICT_DETECTED, ...))
            else:
                triggers.append(Trigger(type=TriggerType.DECISION_FORMING, ...))
        
        # 3. Check for topic drift
        if self._detect_drift(context):
            triggers.append(Trigger(type=TriggerType.TOPIC_DRIFT, ...))
        
        return triggers
```

### 5. LLM Reasoning (`src/reasoning/`)

**Purpose:** Generate appropriate chat messages

**Input Structure:**
```python
@dataclass
class ReasoningRequest:
    trigger: Trigger
    recent_transcript: str      # Last 2-3 minutes
    compressed_context: str     # Summary + state
    bot_config: BotConfig
```

**System Prompt:**
```
You are {bot_name}, a silent AI assistant in a live meeting.

CONTEXT:
{compressed_context}

RECENT CONVERSATION:
{recent_transcript}

TRIGGER: {trigger_type}
TRIGGER REASON: {trigger_reason}

Generate a SHORT chat message (1-3 lines, max 280 chars) that:
- Is neutral and factual
- Is suggestive, not commanding  
- Uses no corporate jargon
- Can be easily ignored if not needed

Format: Just the message text, no labels.
```

**Output Constraints:**
- Max 3 lines
- Max 280 characters
- No markdown formatting
- No bullet points
- No "I think" or "I suggest"

### 6. Chat Manager (`src/chat/`)

**Purpose:** Handle message formatting and rate limiting

**Rate Limiting Rules:**
- Min 30 seconds between proactive messages
- No limit on reactive (command) responses
- 5 second cooldown after any user message
- Max 1 proactive message per topic

```python
class ChatManager:
    async def send_if_allowed(
        self, 
        message: str, 
        is_proactive: bool
    ) -> bool:
        # Check rate limits
        if is_proactive and not self._can_send_proactive():
            return False
        
        # Format message
        formatted = f"{self.config.bot_label}: {message}"
        
        # Send via bot
        await self.bot.send_chat(formatted)
        
        # Update state
        self._last_message_time = datetime.now()
        return True
```

---

## 🔄 State Machine

```
                    ┌──────────────┐
                    │   JOINING    │
                    └──────┬───────┘
                           │ connected
                           ▼
┌─────────┐         ┌──────────────┐
│ LEAVING │◄────────│   LISTENING  │◄─────────────┐
└─────────┘  leave  └──────┬───────┘              │
                           │ trigger fired        │
                           ▼                      │
                    ┌──────────────┐              │
                    │  PROCESSING  │              │
                    └──────┬───────┘              │
                           │ message ready        │
                           ▼                      │
                    ┌──────────────┐              │
                    │   SENDING    │──────────────┘
                    └──────────────┘  sent/skipped
```

---

## 📦 Data Flow Example

### Scenario: Conflict Detection

**Meeting conversation:**
```
[5 min ago]
Sarah: "I'm worried about latency. Our current system has 200ms response times."

[Now]
John: "Let's just use Kafka - it'll solve our scaling problems."
```

**System flow:**

1. **TranscriptChunk** received: "Let's just use Kafka..."
2. **ContextEngine** updates:
   - Adds to sliding window
   - Detects decision language: "Let's just use"
   - Updates `decisions_pending`: Kafka for scaling
3. **TriggerEngine** checks:
   - Decision forming: ✓
   - Cross-reference with `concerns_raised`
   - Found: "latency concerns" from Sarah
   - **Trigger: CONFLICT_DETECTED**
4. **LLMService** generates:
   ```
   Input context: "Sarah raised latency concerns (200ms). Now Kafka suggested."
   Trigger: CONFLICT_DETECTED
   Output: "Earlier latency concerns may conflict with Kafka unless managed services are used."
   ```
5. **ChatManager** sends:
   ```
   Alex (AI): Earlier latency concerns may conflict with Kafka unless managed services are used.
   ```

---

## 🛠️ Technology Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Runtime | Python 3.11+ | Async support, ML ecosystem |
| API Framework | FastAPI + uvicorn | Async, auto-docs, SSE support |
| Meeting Bot | Recall.ai managed API | No browser, handles Teams/Zoom/Meet |
| Real-Time STT | Recall.ai `recallai_streaming` | 1–3 s latency, WebSocket push |
| LLM | OpenAI GPT-4o | Fast, capable |
| TTS (optional) | ElevenLabs | Voice responses in meeting |
| Logging | structlog | Structured JSON in prod, pretty in dev |
| Frontend | Vite 5 + React 18 + Tailwind CSS | Modern SPA |

---

## 🚀 Production Deployment Architecture

```
                         Internet
                             │
                    ┌────────▼────────┐
                    │  Cloudflare DNS  │
                    │  taskora.in.net  │
                    │ api.taskora.in.net│
                    └────────┬────────┘
                             │  :80 / :443
                    ┌────────▼────────┐
                    │  nginx 1.27     │       Oracle Cloud VM
                    │  (SSL termination│       Ubuntu 22.04
                    │   + reverse proxy│       IP: 92.4.69.7
                    └──┬──────────┬───┘
                       │          │
              :3000     │          │  :8000 (REST/SSE)
        ┌──────▼──┐    │    ┌─────▼──────────────┐
        │  React  │    │    │   FastAPI (uvicorn)  │
        │  SPA    │    │    │   src/api/app.py     │
        │  (web)  │    │    │   port 8000          │
        └─────────┘    │    └─────────┬────────────┘
                       │              │  asyncio
                       │    ┌─────────▼────────────┐
                       │    │   MeetingAssistant    │
                       │    │   src/main.py         │
                       │    │                       │
                       │    │ ┌────────────────────┐│
                       │    │ │  RecallBot         ││
                       │    │ │  → Recall.ai API   ││
                       │    │ └────────────────────┘│
                       │    │ ┌────────────────────┐│
                       │    │ │  ContextEngine     ││
                       │    │ │  TriggerEngine     ││
                       │    │ │  LLMService        ││
                       │    │ │  ChatManager       ││
                       │    │ └────────────────────┘│
                       │    └──────────┬────────────┘
                       │               │  WebSocket (port 8766)
                       │    ┌──────────▼────────────┐
                       │    │ RealtimeTranscriptServer│
                       └────│ wss://api.taskora.in.net│◄── Recall.ai cloud
                            │ /realtime               │    pushes transcript
                            └─────────────────────────┘    chunks here
```

### Container Layout (Docker Compose v1.29)

| Container | Image | Ports | Role |
|---|---|---|---|
| `meetmind-nginx` | `nginx:1.27-alpine` | 80, 443 | SSL termination, reverse proxy |
| `meetmind-api` | `ghcr.io/*/meet_ai_agent:latest` | 127.0.0.1:8000, 0.0.0.0:8766 | FastAPI + WebSocket transcript server |
| `meetmind-web` | `ghcr.io/*/meetmind-website:latest` | 127.0.0.1:3000 | React SPA (served as static files) |
| `meetmind-certbot` | `certbot/certbot` | — | Let's Encrypt cert renewal |

### nginx Routing

| Host | Path | Proxies To | Notes |
|---|---|---|---|
| `taskora.in.net` | `/` | `web:80` | React SPA |
| `api.taskora.in.net` | `/` | `api:8000` | FastAPI REST + SSE; buffering disabled for SSE |
| `api.taskora.in.net` | `/realtime` | `api:8766` | WebSocket upgrade for real-time transcript |

### Required Environment Variables (`/opt/meetmind/.env` on VM)

```env
# Auth
API_SECRET_KEY=<random 32-byte token>

# CORS
ALLOWED_ORIGINS=https://taskora.in.net,https://www.taskora.in.net,https://api.taskora.in.net

# Recall.ai
RECALL_API_KEY=<from recall.ai dashboard>
RECALL_REALTIME_URL=wss://api.taskora.in.net/realtime   # ← CRITICAL for live transcript

# OpenAI
OPENAI_API_KEY=<from openai.com>

# (Optional) ElevenLabs TTS
ELEVENLABS_API_KEY=<from elevenlabs.io>

# Docker image org
GITHUB_ORG=adarshbhardwaj27
```

---

## 🔄 CI/CD Pipeline

Both repos use GitHub Actions → GitHub Container Registry → SSH deploy.

```
Developer pushes to branch
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              GitHub Actions                              │
│                                                         │
│  1. checkout code                                       │
│  2. docker/login-action  → ghcr.io (PAT secret)        │
│  3. docker/build-push-action                            │
│       • backend:  no build args                         │
│       • frontend: VITE_API_URL, VITE_API_KEY baked in   │
│  4. SSH into Oracle VM  (secrets: HOST, USER, SSH_KEY)  │
│  5. docker-compose pull <service>                       │
│     docker-compose stop <service>                       │
│     docker-compose rm -f <service>                      │
│     docker-compose up -d --no-deps <service>            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   Oracle Cloud VM
   /opt/meetmind/
   ├── docker-compose.yml
   ├── .env                 ← secrets, never committed
   └── nginx/
       └── nginx.conf
```

### Repository → Branch → Service

| Repo | Trigger Branch | Service Deployed |
|---|---|---|
| `ai_meeting_agent` | `teams-bot-ml-version`, `main` | `meetmind-api` |
| `frontend_ai_meeting_agent` | `master` | `meetmind-web` |

### GitHub Secrets Required

| Secret | Used In |
|---|---|
| `ORACLE_VM_HOST` | Both repos — SSH target IP |
| `ORACLE_VM_USER` | Both repos — SSH username |
| `ORACLE_VM_SSH_KEY` | Both repos — private key (PEM) |
| `CR_PAT` | Both repos — ghcr.io push/pull token |
| `VITE_API_KEY` | Frontend only — baked into React bundle at build time |

---

## 🌐 Real-Time Transcript Flow (Recall.ai)

```
Meeting audio
     │
     ▼
Recall.ai cloud  ──(WebSocket)──►  wss://api.taskora.in.net/realtime
                                          │ nginx proxies to api:8766
                                          ▼
                              RealtimeTranscriptServer (port 8766)
                                          │ _on_transcript_chunk callback
                                          ▼
                              ContextEngine.add_transcript_chunk()
                                          │
                                          ▼
                              TriggerEngine.check_for_triggers()
                               ├── VOICE_ADDRESS ("Alex, …")
                               ├── EXPLICIT_COMMAND ("@Alex summarize")
                               ├── DECISION_FORMING
                               ├── CONFLICT_DETECTED
                               └── … other trigger types
                                          │ trigger fired
                                          ▼
                              LLMService.generate_response()
                                          │
                                          ▼
                              ChatManager → RecallBot.send_chat()
                                          │
                                          ▼
                              Message appears in meeting chat
```

**Without `RECALL_REALTIME_URL` set:** Recall.ai never pushes transcript chunks during the meeting. The fallback `_monitor_transcript` only downloads the full transcript after the recording ends. The trigger engine never fires during the live call.

---

## 📋 Next Steps (TODOs)

### Phase 1: Core Infrastructure ✅
- [x] Context engine with sliding window
- [x] Trigger detection patterns (VOICE_ADDRESS, DECISION_FORMING, CONFLICT_DETECTED, …)
- [x] LLM service wrapper (OpenAI GPT-4o)
- [x] FastAPI layer with SSE log streaming

### Phase 2: Platform Integration ✅
- [x] Recall.ai managed bot (Teams, Zoom, Google Meet)
- [x] Real-time WebSocket transcript server
- [x] End-to-end audio → trigger → chat flow

### Phase 3: Intelligence ✅
- [x] Voice mention detection ("Alex, can you…")
- [x] Conflict & decision pattern matching
- [x] Topic drift detection
- [x] Fast local trigger model (pre-LLM scorer)

### Phase 4: Production ✅
- [x] Docker + docker-compose deployment
- [x] nginx SSL termination (Let's Encrypt)
- [x] GitHub Actions CI/CD
- [x] X-API-Key authentication on all routes
- [x] React dashboard with real API integration + SSE log viewer
