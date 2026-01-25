"""Meeting Bot package."""

from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.bot.google_meet_bot import GoogleMeetBot
from src.bot.teams_bot import TeamsBot
from src.bot.zoom_bot import ZoomBot


def create_bot(platform: str, bot_name: str = None) -> BaseMeetingBot:
    """
    Factory function to create the appropriate bot for a platform.
    
    Args:
        platform: One of 'google_meet', 'teams', 'zoom'
        bot_name: Name for the bot (default: from config)
    
    Returns:
        Bot instance for the specified platform
    """
    platform_lower = platform.lower().replace(" ", "_").replace("-", "_")
    
    if platform_lower in ("google_meet", "meet", "google"):
        return GoogleMeetBot(bot_name)
    elif platform_lower in ("teams", "microsoft_teams", "ms_teams"):
        return TeamsBot(bot_name)
    elif platform_lower in ("zoom",):
        return ZoomBot(bot_name)
    else:
        raise ValueError(f"Unknown platform: {platform}")


def detect_platform(url: str) -> MeetingPlatform:
    """
    Detect the meeting platform from a URL.
    
    Args:
        url: Meeting URL
    
    Returns:
        Detected platform
    """
    url_lower = url.lower()
    
    if "meet.google.com" in url_lower:
        return MeetingPlatform.GOOGLE_MEET
    elif "teams.microsoft.com" in url_lower or "teams.live.com" in url_lower:
        return MeetingPlatform.MICROSOFT_TEAMS
    elif "zoom.us" in url_lower or "zoom.com" in url_lower:
        return MeetingPlatform.ZOOM
    else:
        raise ValueError(f"Could not detect platform from URL: {url}")


__all__ = [
    "BaseMeetingBot",
    "BotState",
    "MeetingPlatform",
    "GoogleMeetBot",
    "TeamsBot",
    "ZoomBot",
    "create_bot",
    "detect_platform",
]
