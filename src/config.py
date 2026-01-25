"""
Configuration management for the Real-Time AI Meeting Assistant.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class STTProvider(Enum):
    DEEPGRAM = "deepgram"
    GOOGLE = "google"
    AZURE = "azure"


class ConflictSensitivity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class BotConfig:
    """Bot identity configuration."""
    name: str = field(default_factory=lambda: os.getenv("BOT_NAME", "Alex"))
    label: str = field(default_factory=lambda: os.getenv("BOT_LABEL", "Alex (AI)"))


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: LLMProvider = field(
        default_factory=lambda: LLMProvider(os.getenv("LLM_PROVIDER", "openai"))
    )
    # OpenAI
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    )
    # Anthropic
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    )
    # Azure OpenAI
    azure_openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY")
    )
    azure_openai_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    azure_openai_deployment: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )


@dataclass
class STTConfig:
    """Speech-to-text configuration."""
    provider: STTProvider = field(
        default_factory=lambda: STTProvider(os.getenv("STT_PROVIDER", "deepgram"))
    )
    deepgram_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("DEEPGRAM_API_KEY")
    )
    google_credentials_path: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    azure_speech_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_SPEECH_KEY")
    )
    azure_speech_region: str = field(
        default_factory=lambda: os.getenv("AZURE_SPEECH_REGION", "eastus")
    )


@dataclass
class ContextConfig:
    """Context engine configuration."""
    window_minutes: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_WINDOW_MINUTES", "10"))
    )
    update_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv("CONTEXT_UPDATE_INTERVAL_SECONDS", "5"))
    )
    max_transcript_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TRANSCRIPT_TOKENS", "4000"))
    )
    summary_update_threshold: int = field(
        default_factory=lambda: int(os.getenv("SUMMARY_UPDATE_THRESHOLD", "500"))
    )


@dataclass
class TriggerConfig:
    """Trigger engine configuration."""
    check_interval_ms: int = field(
        default_factory=lambda: int(os.getenv("TRIGGER_CHECK_INTERVAL_MS", "500"))
    )
    decision_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("DECISION_CONFIDENCE_THRESHOLD", "0.7"))
    )
    conflict_sensitivity: ConflictSensitivity = field(
        default_factory=lambda: ConflictSensitivity(
            os.getenv("CONFLICT_SENSITIVITY", "medium")
        )
    )
    enable_proactive: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PROACTIVE_MESSAGES", "true").lower() == "true"
    )


@dataclass
class ChatConfig:
    """Chat output configuration."""
    max_lines: int = field(
        default_factory=lambda: int(os.getenv("MAX_MESSAGE_LINES", "3"))
    )
    max_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_MESSAGE_CHARS", "280"))
    )
    min_seconds_between_proactive: int = field(
        default_factory=lambda: int(os.getenv("MIN_SECONDS_BETWEEN_PROACTIVE", "30"))
    )
    cooldown_after_user_message: int = field(
        default_factory=lambda: int(os.getenv("COOLDOWN_AFTER_USER_MESSAGE", "5"))
    )


@dataclass
class AppConfig:
    """Main application configuration."""
    bot: BotConfig = field(default_factory=BotConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


# Global config instance
config = AppConfig()
