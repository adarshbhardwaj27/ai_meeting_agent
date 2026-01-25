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
| Async Framework | asyncio + aiohttp | Event-driven architecture |
| STT | Deepgram | Best real-time latency |
| LLM | GPT-4 Turbo / Claude 3 | Fast, capable |
| Meeting Bots | Puppeteer (Meet), Bot Framework (Teams) | Platform-specific |
| Logging | structlog | Structured, async-friendly |

---

## 🚀 Deployment Architecture

### Development
```
Local Machine
├── Python venv
├── ngrok (for webhooks)
└── Test meeting room
```

### Production
```
┌─────────────────────────────────────────────────────┐
│                    Kubernetes                        │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Bot Pod 1  │  │  Bot Pod 2  │  │  Bot Pod N  │ │
│  │  (Meeting1) │  │  (Meeting2) │  │  (MeetingN) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│         │               │               │          │
│         └───────────────┼───────────────┘          │
│                         ▼                          │
│              ┌─────────────────┐                   │
│              │  Redis PubSub   │                   │
│              │  (Event Bus)    │                   │
│              └─────────────────┘                   │
│                         │                          │
│         ┌───────────────┼───────────────┐          │
│         ▼               ▼               ▼          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐   │
│  │ LLM Service │  │ STT Service │  │ Metrics  │   │
│  └─────────────┘  └─────────────┘  └──────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Next Steps (TODOs)

### Phase 1: Core Infrastructure
- [ ] Implement base event bus
- [ ] Implement context engine with sliding window
- [ ] Implement trigger detection patterns
- [ ] Implement LLM service wrapper

### Phase 2: Platform Integration
- [ ] Google Meet bot (Puppeteer + extension)
- [ ] Deepgram streaming integration
- [ ] End-to-end audio → chat flow

### Phase 3: Intelligence
- [ ] Fine-tune trigger sensitivity
- [ ] Implement conflict detection logic
- [ ] Add topic drift detection
- [ ] Optimize LLM prompts for brevity

### Phase 4: Production
- [ ] Add Teams bot support
- [ ] Add Zoom bot support
- [ ] Kubernetes deployment
- [ ] Monitoring & alerting
