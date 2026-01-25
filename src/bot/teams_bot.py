"""
Microsoft Teams Bot implementation.

This bot joins Microsoft Teams meetings using the Microsoft Bot Framework
and Graph API to participate in meetings and interact with chat.

IMPORTANT: Production use requires:
1. Azure Bot Service registration
2. Microsoft 365 admin consent
3. Proper API permissions (Calls.JoinGroupCall.All, etc.)
"""

import asyncio
from typing import Optional
import structlog

from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.config import config

logger = structlog.get_logger()


class TeamsBot(BaseMeetingBot):
    """
    Bot for joining Microsoft Teams meetings.
    
    Uses Microsoft Graph API and Bot Framework to:
    - Join meetings as a bot participant
    - Capture audio via media platform
    - Send chat messages via Graph API
    
    Requirements:
    - Azure Bot registration
    - Microsoft Graph API permissions
    - MSAL authentication
    """
    
    def __init__(self, bot_name: str = None):
        super().__init__(bot_name or config.bot.name)
        self._access_token: Optional[str] = None
        self._call_id: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._audio_socket = None
    
    @property
    def platform(self) -> MeetingPlatform:
        return MeetingPlatform.MICROSOFT_TEAMS
    
    async def join(self, meeting_url: str) -> bool:
        """
        Join a Microsoft Teams meeting.
        
        Args:
            meeting_url: Teams meeting URL or join URL
        
        Returns:
            True if successfully joined
        """
        self._set_state(BotState.CONNECTING)
        self.meeting_url = meeting_url
        
        try:
            # Authenticate with MSAL
            await self._authenticate()
            
            # Parse meeting info from URL
            meeting_info = self._parse_meeting_url(meeting_url)
            
            # Join the call using Graph API
            self._call_id = await self._join_call(meeting_info)
            
            if self._call_id:
                self._set_state(BotState.IN_MEETING)
                
                # Get chat thread ID for this meeting
                self._chat_id = await self._get_meeting_chat_id()
                
                logger.info("teams_meeting_joined", call_id=self._call_id)
                return True
            else:
                self._set_state(BotState.ERROR)
                return False
                
        except Exception as e:
            logger.error("teams_join_failed", error=str(e))
            self._set_state(BotState.ERROR)
            return False
    
    async def leave(self):
        """Leave the Teams meeting."""
        self._set_state(BotState.LEAVING)
        
        try:
            if self._call_id and self._access_token:
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {self._access_token}"}
                    url = f"https://graph.microsoft.com/v1.0/communications/calls/{self._call_id}"
                    
                    async with session.delete(url, headers=headers) as response:
                        if response.status in (200, 204):
                            logger.info("teams_call_ended")
                        else:
                            logger.warning("teams_leave_failed", status=response.status)
            
            self._set_state(BotState.DISCONNECTED)
            
        except Exception as e:
            logger.error("teams_leave_error", error=str(e))
            self._set_state(BotState.DISCONNECTED)
    
    async def send_chat(self, message: str) -> bool:
        """
        Send a chat message in the Teams meeting.
        
        Args:
            message: The message to send
        
        Returns:
            True if sent successfully
        """
        if not self._chat_id or not self._access_token:
            logger.warning("cannot_send_chat", reason="no chat_id or token")
            return False
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "application/json"
                }
                
                url = f"https://graph.microsoft.com/v1.0/chats/{self._chat_id}/messages"
                
                body = {
                    "body": {
                        "content": message,
                        "contentType": "text"
                    }
                }
                
                async with session.post(url, headers=headers, json=body) as response:
                    if response.status == 201:
                        logger.info("teams_chat_sent", message=message[:50])
                        return True
                    else:
                        error = await response.text()
                        logger.error("teams_chat_failed", status=response.status, error=error)
                        return False
                        
        except Exception as e:
            logger.error("teams_send_chat_error", error=str(e))
            return False
    
    async def start_audio_capture(self):
        """
        Start capturing audio from the Teams meeting.
        
        This requires the bot to be registered as a media bot
        with appropriate permissions.
        """
        # Teams media bots require special setup with media platform
        # This is a placeholder for the actual implementation
        logger.info("teams_audio_capture_started")
        
        # In production, you would:
        # 1. Configure media platform (Azure Media Services or self-hosted)
        # 2. Subscribe to audio streams via Graph API
        # 3. Process raw audio from the media platform
        pass
    
    async def stop_audio_capture(self):
        """Stop audio capture."""
        if self._audio_socket:
            self._audio_socket.close()
        logger.info("teams_audio_capture_stopped")
    
    async def _authenticate(self):
        """Authenticate with Microsoft identity platform using MSAL."""
        try:
            from msal import ConfidentialClientApplication
            
            app = ConfidentialClientApplication(
                client_id=config.llm.azure_openai_api_key,  # Reusing config structure
                authority=f"https://login.microsoftonline.com/{config.llm.azure_openai_endpoint}",
                client_credential=config.llm.azure_openai_deployment
            )
            
            # Get token for Graph API
            result = app.acquire_token_for_client(
                scopes=["https://graph.microsoft.com/.default"]
            )
            
            if "access_token" in result:
                self._access_token = result["access_token"]
                logger.info("teams_authenticated")
            else:
                raise Exception(f"Authentication failed: {result.get('error_description')}")
                
        except ImportError:
            logger.error("msal_not_installed", hint="Run: pip install msal")
            raise
        except Exception as e:
            logger.error("teams_auth_failed", error=str(e))
            raise
    
    def _parse_meeting_url(self, url: str) -> dict:
        """Parse Teams meeting URL to extract meeting info."""
        # Teams URLs contain encoded meeting info
        # Example: https://teams.microsoft.com/l/meetup-join/...
        
        import urllib.parse
        
        parsed = urllib.parse.urlparse(url)
        
        # Extract meeting details from path/query
        meeting_info = {
            "thread_id": None,
            "organizer_id": None,
            "tenant_id": None,
            "message_id": None,
        }
        
        # Parse the URL path for meeting details
        if "meetup-join" in parsed.path:
            path_parts = parsed.path.split("/")
            for i, part in enumerate(path_parts):
                if part == "meetup-join" and i + 1 < len(path_parts):
                    # The next part is the encoded meeting info
                    encoded_info = path_parts[i + 1]
                    # Decode and parse
                    decoded = urllib.parse.unquote(encoded_info)
                    # Parse the decoded info
                    # Format: 19:meeting_xxx@thread.v2/0?context={"Tid":"...","Oid":"..."}
                    if "@thread" in decoded:
                        meeting_info["thread_id"] = decoded.split("/")[0]
        
        # Parse query parameters for context
        query_params = urllib.parse.parse_qs(parsed.query)
        if "context" in query_params:
            import json
            context = json.loads(query_params["context"][0])
            meeting_info["tenant_id"] = context.get("Tid")
            meeting_info["organizer_id"] = context.get("Oid")
        
        return meeting_info
    
    async def _join_call(self, meeting_info: dict) -> Optional[str]:
        """Join the Teams call using Graph API."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "application/json"
                }
                
                # Join call endpoint
                url = "https://graph.microsoft.com/v1.0/communications/calls"
                
                body = {
                    "@odata.type": "#microsoft.graph.call",
                    "callbackUri": "https://your-callback-url.com/callback",  # Your callback endpoint
                    "chatInfo": {
                        "@odata.type": "#microsoft.graph.chatInfo",
                        "threadId": meeting_info.get("thread_id"),
                        "messageId": "0"
                    },
                    "meetingInfo": {
                        "@odata.type": "#microsoft.graph.organizerMeetingInfo",
                        "organizer": {
                            "@odata.type": "#microsoft.graph.identitySet",
                            "user": {
                                "@odata.type": "#microsoft.graph.identity",
                                "id": meeting_info.get("organizer_id"),
                                "tenantId": meeting_info.get("tenant_id")
                            }
                        }
                    },
                    "mediaConfig": {
                        "@odata.type": "#microsoft.graph.appHostedMediaConfig",
                        "blob": "<media config blob>"
                    },
                    "tenantId": meeting_info.get("tenant_id")
                }
                
                async with session.post(url, headers=headers, json=body) as response:
                    if response.status == 201:
                        data = await response.json()
                        return data.get("id")
                    else:
                        error = await response.text()
                        logger.error("teams_join_call_failed", status=response.status, error=error)
                        return None
                        
        except Exception as e:
            logger.error("teams_join_call_error", error=str(e))
            return None
    
    async def _get_meeting_chat_id(self) -> Optional[str]:
        """Get the chat ID for the meeting."""
        if not self._call_id:
            return None
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self._access_token}"}
                
                # Get call details which includes chat info
                url = f"https://graph.microsoft.com/v1.0/communications/calls/{self._call_id}"
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        chat_info = data.get("chatInfo", {})
                        return chat_info.get("threadId")
                    else:
                        return None
                        
        except Exception as e:
            logger.error("get_chat_id_error", error=str(e))
            return None
