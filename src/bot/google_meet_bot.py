"""
Google Meet Bot implementation.

This bot joins Google Meet meetings using browser automation (Playwright)
to capture audio and interact with the meeting chat.

APPROACH: Uses a persistent browser profile so you only need to sign in ONCE.
After the first sign-in, the bot remembers your session and can join meetings
automatically without any manual intervention.

FIRST-TIME SETUP:
1. Run the bot once - a browser will open
2. Sign in to your Google account manually
3. Close the browser (or let the bot close it)
4. Future runs will use the saved session - no sign-in needed!
"""

import asyncio
import os
from typing import Optional
from pathlib import Path
import structlog

from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.config import config

logger = structlog.get_logger()

# Persistent browser data directory - stores cookies/session
BROWSER_DATA_DIR = Path(__file__).parent.parent.parent / ".browser_data"


class GoogleMeetBot(BaseMeetingBot):
    """
    Bot for joining Google Meet meetings.
    
    Uses browser automation with PERSISTENT PROFILE to:
    - Join meetings as a participant (no repeated sign-in)
    - Capture audio stream
    - Read and send chat messages
    
    The browser profile is saved to .browser_data/ folder.
    Sign in once, and the bot remembers your session forever.
    """
    
    def __init__(self, bot_name: str = None):
        super().__init__(bot_name or config.bot.name)
        self._playwright = None
        self._browser = None
        self._page = None
        self._chat_monitor_task = None
    
    @property
    def platform(self) -> MeetingPlatform:
        return MeetingPlatform.GOOGLE_MEET
    
    def get_page(self):
        """Return the Playwright page object for caption scraping."""
        return self._page
    
    async def join(self, meeting_url: str) -> bool:
        """
        Join a Google Meet meeting.
        
        Uses a PERSISTENT browser profile - sign in once, join forever.
        
        Args:
            meeting_url: Google Meet URL (e.g., https://meet.google.com/xxx-xxxx-xxx)
        
        Returns:
            True if successfully joined
        """
        self._set_state(BotState.CONNECTING)
        self.meeting_url = meeting_url
        
        try:
            from playwright.async_api import async_playwright
            
            self._playwright = await async_playwright().start()
            
            # Ensure browser data directory exists
            BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Use PERSISTENT context - this saves cookies, localStorage, session!
            # After signing in once, you never need to sign in again.
            logger.info("launching_browser", 
                       profile_dir=str(BROWSER_DATA_DIR),
                       hint="First run? Sign in to Google when browser opens. Future runs will be automatic!")
            
            # Launch Chromium with persistent context
            # Google Meet needs a real browser (WebRTC doesn't work headless)
            # but we minimize the window so it stays out of your way.
            self._browser = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(BROWSER_DATA_DIR),
                headless=False,  # Required — Google Meet blocks headless
                channel="chrome",
                args=[
                    '--use-fake-ui-for-media-stream',
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--window-size=800,600',      # Small window
                    '--window-position=10000,0',   # Move off-screen initially
                ],
                ignore_default_args=['--enable-automation'],
            )
            
            # Get the page (persistent context already has one)
            self._page = self._browser.pages[0] if self._browser.pages else await self._browser.new_page()
            
            # Navigate to Google Meet
            logger.info("navigating_to_meet", url=meeting_url)
            await self._page.goto(meeting_url, wait_until='networkidle')
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Wait a bit more for page to fully load
            await asyncio.sleep(2)
            
            # Check if we're on the Google sign-in page (not Meet page)
            needs_signin = await self._check_needs_signin()
            if needs_signin:
                logger.warning("google_signin_required",
                             message="Please sign in to Google in the browser window.",
                             hint="This is a ONE-TIME setup. After signing in, close this message and the bot will remember your session.")
                # Wait for user to sign in (up to 5 minutes)
                await self._wait_for_signin(timeout=300)
            
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
                logger.info("successfully_joined_meeting", url=meeting_url)
                # Minimize the browser window — bot runs silently in background
                try:
                    cdp = await self._page.context.new_cdp_session(self._page)
                    await cdp.send('Browser.getWindowForTarget')
                    window = await cdp.send('Browser.getWindowForTarget')
                    await cdp.send('Browser.setWindowBounds', {
                        'windowId': window['windowId'],
                        'bounds': {'windowState': 'minimized'}
                    })
                    print("🔽 Browser minimized — bot running in background")
                except Exception:
                    # Fallback: just log that we couldn't minimize
                    print("💡 Tip: You can manually minimize the Chrome window")
                # Start monitoring chat
                self._chat_monitor_task = asyncio.create_task(self._monitor_chat())
                return True
            else:
                self._set_state(BotState.ERROR)
                return False
                
        except ImportError:
            logger.error("playwright_not_installed", 
                        hint="Run: pip install playwright && playwright install chromium")
            self._set_state(BotState.ERROR)
            return False
        except Exception as e:
            logger.error("meet_join_failed", error=str(e))
            self._set_state(BotState.ERROR)
            return False
    
    async def _check_needs_signin(self) -> bool:
        """Check if we need to sign in to Google (actually on login page, not just seeing 'Sign in' button)."""
        try:
            current_url = self._page.url
            
            # If we're on accounts.google.com, we definitely need to sign in
            if 'accounts.google.com' in current_url:
                return True
            
            # If we're on meet.google.com, check if it's asking to sign in
            # Look for the actual sign-in page elements, not just a "Sign in" link
            if 'meet.google.com' in current_url:
                # Check for "Your name" input - this means we can join without signing in
                name_input = await self._page.query_selector('input[aria-label="Your name"]')
                if name_input:
                    return False  # We can join as guest
                
                # Check for join buttons - if they exist, we're already signed in
                join_button = await self._page.query_selector('[aria-label="Join now"], [aria-label="Ask to join"]')
                if join_button:
                    return False  # Already signed in or can join
                
                # Check for camera/mic toggle - means we're on the pre-join screen
                media_toggle = await self._page.query_selector('[aria-label*="camera" i], [aria-label*="microphone" i]')
                if media_toggle:
                    return False  # On pre-join screen, good to go
            
            # Check for explicit sign-in page elements
            email_input = await self._page.query_selector('input[type="email"][name="identifier"]')
            if email_input:
                return True  # On Google sign-in page
            
            return False
        except:
            return False
    
    async def _wait_for_signin(self, timeout: int = 300):
        """Wait for user to complete Google sign-in."""
        logger.info("waiting_for_signin", timeout_seconds=timeout)
        
        for i in range(timeout):
            # Check if we're now on the meeting page (sign-in complete)
            needs_signin = await self._check_needs_signin()
            if not needs_signin:
                logger.info("signin_complete", message="Google sign-in successful! Session saved for future runs.")
                await asyncio.sleep(2)  # Let page settle
                return
            
            if i % 30 == 0 and i > 0:
                logger.info("still_waiting_for_signin", elapsed_seconds=i)
            
            await asyncio.sleep(1)
        
        raise Exception("Sign-in timeout - please sign in faster next time!")
    
    async def leave(self):
        """Leave the Google Meet meeting."""
        self._set_state(BotState.LEAVING)
        
        try:
            # Stop background tasks
            if self._chat_monitor_task:
                self._chat_monitor_task.cancel()
            
            # Click leave button
            if self._page:
                try:
                    leave_button = await self._page.query_selector('[aria-label="Leave call"]')
                    if leave_button:
                        await leave_button.click()
                except:
                    pass
            
            # Close browser context (this saves the session!)
            if self._browser:
                await self._browser.close()
            
            # Stop playwright
            if self._playwright:
                await self._playwright.stop()
            
            self._set_state(BotState.DISCONNECTED)
            logger.info("left_meeting", message="Session saved - no sign-in needed next time!")
            
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
        """
        Monitor the Google Meet chat for new messages.

        IMPORTANT: The chat panel must be open for message elements to
        exist in the DOM.  We open it once on startup and then poll.
        """
        # ── Step 1: Open the chat panel ─────────────────────────────
        await asyncio.sleep(3)  # Let meeting UI settle

        chat_opened = False
        try:
            chat_btn_selectors = [
                '[aria-label="Chat with everyone"]',
                'button[aria-label*="Chat" i]',
                '[data-panel-id="2"]',
            ]
            for selector in chat_btn_selectors:
                try:
                    btn = await self._page.query_selector(selector)
                    if btn:
                        await btn.click()
                        chat_opened = True
                        print("💬 Chat panel opened for monitoring")
                        logger.info("chat_panel_opened", selector=selector)
                        await asyncio.sleep(1)
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning("chat_panel_open_error", error=str(e))

        if not chat_opened:
            print("⚠️  Could not open chat panel — @Alex commands won't work")
            logger.warning("chat_panel_not_opened")

        # ── Step 2: One-time debug dump of chat elements ────────────
        await self._debug_dump_chat_elements()

        # ── Step 3: Poll for new messages ───────────────────────────
        last_message_count = 0

        while self.state == BotState.IN_MEETING:
            try:
                # Use JS evaluation — more robust than query_selector_all
                result = await self._page.evaluate("""
                    () => {
                        const messages = [];

                        // Strategy 1: data-message-text attribute
                        const msgEls = document.querySelectorAll('[data-message-text]');
                        if (msgEls.length > 0) {
                            for (const el of msgEls) {
                                const text = el.getAttribute('data-message-text')
                                            || el.textContent.trim();
                                let sender = 'Unknown';
                                // Sender info might be on an ancestor or sibling
                                const senderEl = el.closest('[data-sender-name]')
                                              || el.parentElement?.querySelector('[data-sender-name]');
                                if (senderEl) {
                                    sender = senderEl.getAttribute('data-sender-name');
                                }
                                if (text) messages.push({ sender, text });
                            }
                            return { strategy: 1, messages };
                        }

                        // Strategy 2: Locate chat panel via the chat input
                        const chatInput = document.querySelector(
                            'textarea[aria-label="Send a message"], '
                          + 'textarea[aria-label="Send a message to everyone"], '
                          + '[contenteditable="true"][aria-label*="message" i]'
                        );
                        if (!chatInput) return { strategy: 0, messages: [] };

                        // Walk up to find the scrollable messages container
                        let panel = chatInput.parentElement;
                        for (let i = 0; i < 15 && panel; i++) {
                            // The messages container is scrollable and tall
                            if (panel.scrollHeight > 100 &&
                                panel.scrollHeight > panel.clientHeight + 5) {
                                break;
                            }
                            panel = panel.parentElement;
                        }
                        if (!panel) return { strategy: 0, messages: [] };

                        // Collect visible message-like blocks:
                        // Each chat message in Meet is a container div holding
                        // sender-name + message-text children.
                        // We look for divs whose direct text is ≥3 chars and
                        // that aren't buttons / inputs.
                        const candidateDivs = panel.querySelectorAll('div, span');
                        const seen = new Set();

                        for (const el of candidateDivs) {
                            // Only leaf-ish text
                            const text = el.textContent.trim();
                            if (!text || text.length < 2 || text.length > 1000) continue;
                            if (seen.has(text)) continue;

                            // Skip form elements & buttons
                            if (el.closest('button, textarea, input, [role="button"]'))
                                continue;

                            // Skip timestamps
                            if (/^\\d{1,2}:\\d{2}\\s*(AM|PM)?$/i.test(text)) continue;

                            // Must be a leaf (no child div/span with same text)
                            let isLeaf = true;
                            for (const child of el.querySelectorAll('div, span')) {
                                if (child !== el &&
                                    child.textContent.trim() === text) {
                                    isLeaf = false;
                                    break;
                                }
                            }
                            if (!isLeaf) continue;

                            seen.add(text);
                            messages.push({ sender: 'Unknown', text });
                        }

                        return { strategy: 2, messages };
                    }
                """)

                if result:
                    msg_list = result.get("messages", [])
                    strategy = result.get("strategy", 0)

                    if len(msg_list) > last_message_count:
                        for msg in msg_list[last_message_count:]:
                            text = msg.get("text", "")
                            sender = msg.get("sender", "Unknown")
                            if sender != self.bot_name and text:
                                print(f"💬 Chat [{sender}]: {text}")
                                logger.info(
                                    "chat_message_detected",
                                    sender=sender,
                                    text=text[:80],
                                    strategy=strategy,
                                )
                                self._emit_chat_message(sender, text)

                        last_message_count = len(msg_list)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("chat_monitor_error", error=str(e))
                await asyncio.sleep(2)

    async def _debug_dump_chat_elements(self):
        """One-time dump of chat panel elements to help tune selectors."""
        try:
            result = await self._page.evaluate("""
                () => {
                    const info = {};

                    // Check data-message-text elements
                    info.dataMessageTextCount = document.querySelectorAll(
                        '[data-message-text]'
                    ).length;

                    // Check for chat input
                    const chatInput = document.querySelector(
                        'textarea[aria-label="Send a message"], '
                      + 'textarea[aria-label="Send a message to everyone"]'
                    );
                    info.chatInputFound = !!chatInput;

                    // Sample text elements near the chat area
                    if (chatInput) {
                        let panel = chatInput.parentElement;
                        for (let i = 0; i < 10 && panel; i++) {
                            if (panel.getBoundingClientRect().height > 300) break;
                            panel = panel.parentElement;
                        }
                        if (panel) {
                            const texts = [];
                            const els = panel.querySelectorAll('div, span');
                            for (const el of els) {
                                const t = el.textContent.trim();
                                if (t && t.length > 1 && t.length < 100) {
                                    let isLeaf = true;
                                    for (const c of el.querySelectorAll('div, span')) {
                                        if (c !== el && c.textContent.trim() === t) {
                                            isLeaf = false; break;
                                        }
                                    }
                                    if (isLeaf && !texts.includes(t)) {
                                        texts.push(t);
                                    }
                                }
                                if (texts.length >= 15) break;
                            }
                            info.sampleTexts = texts;
                        }
                    }

                    return info;
                }
            """)
            if result:
                print("🔍 Chat panel debug:")
                print(f"   data-message-text elements: {result.get('dataMessageTextCount', 0)}")
                print(f"   Chat input found: {result.get('chatInputFound', False)}")
                samples = result.get("sampleTexts", [])
                if samples:
                    print(f"   Sample texts in chat area: {samples[:8]}")
                logger.info("chat_debug_dump", **result)
        except Exception:
            pass
