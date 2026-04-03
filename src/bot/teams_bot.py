"""
Microsoft Teams Bot implementation.

Uses Recall.ai as the meeting infrastructure — no browser needed.
Recall.ai handles joining the Teams meeting, audio capture, STT,
and chat message delivery. This class is a thin Teams-specific
wrapper around RecallBot.

For Teams (and Zoom), the system uses Recall.ai (managed API).
"""

import asyncio
from typing import Optional
import structlog

from src.bot.recall_bot import RecallBot
from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.config import config

logger = structlog.get_logger()


class TeamsBot(RecallBot):
    """
    Bot for joining Microsoft Teams meetings via Recall.ai.

    Teams runs on a desktop app, so local browser automation is not the
    right integration path. Instead, we use Recall.ai's managed API
    which handles the entire meeting join + audio + transcription flow.

    This class inherits everything from RecallBot and just sets the
    platform to MICROSOFT_TEAMS. Extend this class if Teams-specific
    behavior is needed in the future (e.g., Teams-specific chat
    formatting, adaptive cards, etc.).
    """

    def __init__(self, bot_name: str = None):
        super().__init__(
            bot_name=bot_name,
            platform=MeetingPlatform.MICROSOFT_TEAMS,
        )
        logger.info("teams_bot_initialized", bot_name=self.bot_name)

    # ── Teams-specific overrides (extend as needed) ──────────────────

    async def join(self, meeting_url: str) -> bool:
        """
        Join a Teams meeting.

        Validates the URL looks like a Teams meeting link before
        delegating to Recall.ai.
        """
        # Validate it's actually a Teams URL
        url_lower = meeting_url.lower()
        if "teams.microsoft.com" not in url_lower and "teams.live.com" not in url_lower:
            logger.warning(
                "not_a_teams_url",
                url=meeting_url,
                hint="Expected a teams.microsoft.com or teams.live.com URL",
            )

        print(f"📞 Joining Teams meeting via Recall.ai...")
        return await super().join(meeting_url)
