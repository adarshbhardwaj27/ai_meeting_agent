"""
Google Meet Bot implementation.

This bot joins Google Meet meetings using browser automation (Puppeteer/Playwright)
to capture audio and interact with the meeting chat.

IMPORTANT: This is a simplified implementation. Production use requires:
1. A Google Workspace account with bot permissions
2. Proper OAuth2 authentication
3. Compliance with Google's Terms of Service
"""

import asyncio
from typing import Optional
import structlog

from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.config import config

logger = structlog.get_logger()


class GoogleMeetBot(BaseMeetingBot):
    """
    Bot for joining Google Meet meetings.
    
    Uses browser automation to:
    - Join meetings as a participant
    - Capture audio stream
    - Read and send chat messages
    
    Requirements:
    - Puppeteer or Playwright installed
    - Google account credentials
    - Chrome/Chromium browser
    """
    
    def __init__(self, bot_name: str = None):
        super().__init__(bot_name or config.bot.name)
        self._browser = None
        self._page = None
        self._audio_capture_task = None
        self._chat_monitor_task = None
    
    @property
    def platform(self) -> MeetingPlatform:
        return MeetingPlatform.GOOGLE_MEET
    
    async def join(self, meeting_url: str) -> bool:
        """
        Join a Google Meet meeting.
        
        Args:
            meeting_url: Google Meet URL (e.g., https://meet.google.com/xxx-xxxx-xxx)
        
        Returns:
            True if successfully joined
        """
        self._set_state(BotState.CONNECTING)
        self.meeting_url = meeting_url
        
        try:
            # Launch browser with Playwright
            from playwright.async_api import async_playwright
            
            playwright = await async_playwright().start()
            
            # Launch browser with specific flags for audio capture
            self._browser = await playwright.chromium.launch(
                headless=False,  # Set True for production
                args=[
                    '--use-fake-ui-for-media-stream',
                    '--use-fake-device-for-media-stream',
                    '--disable-web-security',
                    '--allow-running-insecure-content',
                ]
            )
            
            # Create a new page/context
            context = await self._browser.new_context(
                permissions=['microphone', 'camera'],
                user_agent=f'{self.bot_name} Meeting Bot'
            )
            self._page = await context.new_page()
            
            # Navigate to Google Meet
            logger.info("navigating_to_meet", url=meeting_url)
            await self._page.goto(meeting_url)
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Turn off camera and microphone before joining
            await self._disable_media()
            
            # Enter bot name if prompted
            await self._enter_name()
            
            # Click "Ask to join" or "Join now" button
            await self._click_join()
            
            # Wait for admission (host may need to let us in)
            joined = await self._wait_for_admission()
            
            if joined:
                self._set_state(BotState.IN_MEETING)
                # Start monitoring chat
                self._chat_monitor_task = asyncio.create_task(self._monitor_chat())
                return True
            else:
                self._set_state(BotState.ERROR)
                return False
                
        except ImportError:
            logger.error("playwright_not_installed", 
                        hint="Run: pip install playwright && playwright install")
            self._set_state(BotState.ERROR)
            return False
        except Exception as e:
            logger.error("meet_join_failed", error=str(e))
            self._set_state(BotState.ERROR)
            return False
    
    async def leave(self):
        """Leave the Google Meet meeting."""
        self._set_state(BotState.LEAVING)
        
        try:
            # Stop background tasks
            if self._chat_monitor_task:
                self._chat_monitor_task.cancel()
            if self._audio_capture_task:
                self._audio_capture_task.cancel()
            
            # Click leave button
            if self._page:
                try:
                    leave_button = await self._page.query_selector('[aria-label="Leave call"]')
                    if leave_button:
                        await leave_button.click()
                except:
                    pass
            
            # Close browser
            if self._browser:
                await self._browser.close()
            
            self._set_state(BotState.DISCONNECTED)
            logger.info("left_meeting")
            
        except Exception as e:
            logger.error("leave_meeting_error", error=str(e))
            self._set_state(BotState.DISCONNECTED)
    
    async def send_chat(self, message: str) -> bool:
        """
        Send a chat message in Google Meet.
        
        Args:
            message: The message to send
        
        Returns:
            True if sent successfully
        """
        if not self._page or self.state != BotState.IN_MEETING:
            return False
        
        try:
            # Open chat panel if not already open
            chat_button = await self._page.query_selector('[aria-label="Chat with everyone"]')
            if chat_button:
                await chat_button.click()
                await asyncio.sleep(0.5)
            
            # Find chat input
            chat_input = await self._page.query_selector('textarea[aria-label="Send a message"]')
            if not chat_input:
                chat_input = await self._page.query_selector('[data-is-input="true"]')
            
            if chat_input:
                await chat_input.fill(message)
                await chat_input.press('Enter')
                logger.info("chat_sent", message=message[:50])
                return True
            else:
                logger.warning("chat_input_not_found")
                return False
                
        except Exception as e:
            logger.error("send_chat_failed", error=str(e))
            return False
    
    async def start_audio_capture(self):
        """
        Start capturing audio from the meeting.
        
        This uses the Web Audio API to capture the meeting audio stream.
        """
        if not self._page:
            return
        
        try:
            # Inject audio capture script
            await self._page.evaluate('''
                () => {
                    // Create audio context to capture meeting audio
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    
                    // Get all audio elements
                    const audioElements = document.querySelectorAll('audio, video');
                    
                    // Create a media stream destination
                    const destination = audioContext.createMediaStreamDestination();
                    
                    audioElements.forEach(element => {
                        if (element.srcObject) {
                            const source = audioContext.createMediaStreamSource(element.srcObject);
                            source.connect(destination);
                        }
                    });
                    
                    // Set up recorder
                    window._meetingRecorder = new MediaRecorder(destination.stream);
                    window._audioChunks = [];
                    
                    window._meetingRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) {
                            window._audioChunks.push(e.data);
                        }
                    };
                    
                    window._meetingRecorder.start(100); // 100ms chunks
                }
            ''')
            
            self._audio_capture_task = asyncio.create_task(self._audio_capture_loop())
            logger.info("audio_capture_started")
            
        except Exception as e:
            logger.error("audio_capture_start_failed", error=str(e))
    
    async def stop_audio_capture(self):
        """Stop audio capture."""
        if self._audio_capture_task:
            self._audio_capture_task.cancel()
        
        if self._page:
            try:
                await self._page.evaluate('window._meetingRecorder?.stop()')
            except:
                pass
        
        logger.info("audio_capture_stopped")
    
    async def _disable_media(self):
        """Turn off camera and microphone before joining."""
        try:
            # Turn off camera
            camera_button = await self._page.query_selector('[aria-label*="camera" i]')
            if camera_button:
                is_on = await camera_button.get_attribute('data-is-muted')
                if is_on != 'true':
                    await camera_button.click()
            
            # Turn off microphone  
            mic_button = await self._page.query_selector('[aria-label*="microphone" i]')
            if mic_button:
                is_on = await mic_button.get_attribute('data-is-muted')
                if is_on != 'true':
                    await mic_button.click()
                    
        except Exception as e:
            logger.warning("disable_media_failed", error=str(e))
    
    async def _enter_name(self):
        """Enter the bot name if prompted."""
        try:
            name_input = await self._page.query_selector('input[aria-label="Your name"]')
            if name_input:
                await name_input.fill(f"{self.bot_name} (AI)")
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning("enter_name_failed", error=str(e))
    
    async def _click_join(self):
        """Click the join button."""
        try:
            # Try different button selectors
            join_selectors = [
                '[aria-label="Join now"]',
                '[aria-label="Ask to join"]',
                'button:has-text("Join now")',
                'button:has-text("Ask to join")',
            ]
            
            for selector in join_selectors:
                try:
                    button = await self._page.query_selector(selector)
                    if button:
                        await button.click()
                        logger.info("clicked_join_button", selector=selector)
                        return
                except:
                    continue
                    
        except Exception as e:
            logger.warning("click_join_failed", error=str(e))
    
    async def _wait_for_admission(self, timeout: int = 60) -> bool:
        """Wait to be admitted to the meeting."""
        try:
            # Wait for indicators that we're in the meeting
            for _ in range(timeout):
                # Check for meeting controls (indicates we're in)
                controls = await self._page.query_selector('[aria-label="Leave call"]')
                if controls:
                    return True
                
                # Check for rejection
                rejected = await self._page.query_selector('text="You can\'t join this call"')
                if rejected:
                    logger.warning("join_rejected")
                    return False
                
                await asyncio.sleep(1)
            
            logger.warning("admission_timeout")
            return False
            
        except Exception as e:
            logger.error("wait_for_admission_failed", error=str(e))
            return False
    
    async def _monitor_chat(self):
        """Monitor the chat for new messages."""
        last_message_count = 0
        
        while self.state == BotState.IN_MEETING:
            try:
                # Get chat messages
                messages = await self._page.query_selector_all('[data-message-text]')
                
                if len(messages) > last_message_count:
                    # Process new messages
                    for msg in messages[last_message_count:]:
                        try:
                            text = await msg.get_attribute('data-message-text')
                            sender_el = await msg.query_selector('[data-sender-name]')
                            sender = await sender_el.get_attribute('data-sender-name') if sender_el else 'Unknown'
                            
                            if sender != self.bot_name:
                                self._emit_chat_message(sender, text)
                        except:
                            pass
                    
                    last_message_count = len(messages)
                
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("chat_monitor_error", error=str(e))
                await asyncio.sleep(1)
    
    async def _audio_capture_loop(self):
        """Loop to retrieve and emit audio chunks."""
        while self.state == BotState.IN_MEETING:
            try:
                # Get audio chunks from browser
                chunks = await self._page.evaluate('''
                    () => {
                        const chunks = window._audioChunks || [];
                        window._audioChunks = [];
                        return chunks.map(c => Array.from(new Uint8Array(c)));
                    }
                ''')
                
                for chunk_data in chunks:
                    audio_bytes = bytes(chunk_data)
                    if audio_bytes:
                        self._emit_audio_chunk(audio_bytes)
                
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("audio_capture_loop_error", error=str(e))
                await asyncio.sleep(0.5)
