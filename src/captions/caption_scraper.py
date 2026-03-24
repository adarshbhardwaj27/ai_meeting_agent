"""
Google Meet Caption Scraper.

Instead of capturing audio → sending to STT → getting text,
we simply read the captions that Google Meet ALREADY generates.

This is the approach used by browser extensions like Tactiq, MeetGeek, etc.
It's free, reliable, and requires zero audio plumbing.

How it works:
1. After joining a meeting, we enable Google Meet's "Turn on captions" feature
2. Google Meet displays captions as DOM elements in real-time
3. We poll the DOM every 200ms and extract the caption text + speaker name
4. Each new caption line is emitted as a TranscriptChunk
"""

import asyncio
from datetime import datetime
from typing import Callable, Optional, Set
import structlog

from src.context.models import TranscriptChunk

logger = structlog.get_logger()


class CaptionScraper:
    """
    Scrapes Google Meet's built-in live captions from the DOM.

    This completely replaces the audio capture + Deepgram STT pipeline.
    Google does the speech recognition for us — we just read the output.
    """

    def __init__(
        self,
        on_transcript: Optional[Callable[[TranscriptChunk], None]] = None,
        poll_interval_ms: int = 300,
        meeting_url: str = "",
    ):
        self.on_transcript = on_transcript
        self.poll_interval_ms = poll_interval_ms

        self._page = None
        self._running = False
        self._scrape_task: Optional[asyncio.Task] = None
        self._seen_texts: Set[str] = set()
        self._last_caption_text: str = ""
        self._last_caption_speaker: str = ""

        # Extract meeting code for filtering (e.g. "dhq-qotr-evr")
        self._meeting_code = ""
        if meeting_url:
            import re
            match = re.search(r'([a-z]{3}-[a-z]{4}-[a-z]{3})', meeting_url)
            if match:
                self._meeting_code = match.group(1)

    async def start(self, page):
        """
        Start scraping captions from a Google Meet page.

        Args:
            page: Playwright page object (must already be in a meeting)
        """
        self._page = page
        self._running = True

        # Step 1: Enable captions in Google Meet
        await self._enable_captions()

        # Step 2: Start the polling loop
        self._scrape_task = asyncio.create_task(self._scrape_loop())
        print("✅ Caption scraper started — reading Google Meet captions")
        logger.info("caption_scraper_started", poll_ms=self.poll_interval_ms)

    async def stop(self):
        """Stop scraping captions."""
        self._running = False
        if self._scrape_task:
            self._scrape_task.cancel()
            try:
                await self._scrape_task
            except asyncio.CancelledError:
                pass
        logger.info("caption_scraper_stopped")

    # ── Enable captions ──────────────────────────────────────────────

    async def _enable_captions(self):
        """
        Click the "Turn on captions" button in Google Meet.

        Google Meet has a [CC] button in the bottom toolbar.
        We try multiple selectors because Google updates the UI often.
        """
        try:
            # Give the meeting UI a moment to settle
            await asyncio.sleep(3)

            # Strategy 1: keyboard shortcut (c key toggles captions — most reliable)
            try:
                await self._page.keyboard.press("c")
                print("✅ Captions toggled via keyboard shortcut (c)")
                logger.info("captions_enabled_via_keyboard")
                await asyncio.sleep(2)
                return
            except Exception:
                pass

            # Strategy 2: aria-label based selectors
            caption_selectors = [
                'button[aria-label="Turn on captions"]',
                'button[aria-label="Turn on captions (c)"]',
                '[aria-label="Turn on captions"]',
                '[aria-label="Turn on captions (c)"]',
            ]

            for selector in caption_selectors:
                try:
                    btn = await self._page.query_selector(selector)
                    if btn:
                        await btn.click()
                        print("✅ Captions turned ON")
                        logger.info("captions_enabled", selector=selector)
                        await asyncio.sleep(2)
                        return
                except Exception:
                    continue

            logger.warning(
                "captions_enable_failed",
                hint="Could not auto-enable captions. Please press 'c' in the meeting.",
            )
            print(
                "⚠️  Could not auto-enable captions. Press 'c' in the Meet window to turn them on."
            )

        except Exception as e:
            logger.error("captions_enable_error", error=str(e))

    # ── Caption scraping loop ────────────────────────────────────────

    async def _scrape_loop(self):
        """Poll the DOM for caption elements every poll_interval_ms."""
        consecutive_empty = 0
        self._debug_dump_done = False

        while self._running:
            try:
                captions = await self._extract_captions()

                if captions:
                    consecutive_empty = 0
                    for speaker, text in captions:
                        self._emit(speaker, text)
                else:
                    consecutive_empty += 1
                    # Log occasionally if no captions seen for a while
                    if consecutive_empty == 100:  # ~30 seconds
                        print(
                            "⏳ No captions detected yet. Make sure someone is speaking and captions are ON (press 'c' in Meet)."
                        )
                    # After 50 empty polls, do one debug dump to help tune selectors
                    if consecutive_empty == 50 and not self._debug_dump_done:
                        self._debug_dump_done = True
                        await self._debug_dump_bottom_elements()

                await asyncio.sleep(self.poll_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("caption_scrape_error", error=str(e))
                await asyncio.sleep(1)

    async def _debug_dump_bottom_elements(self):
        """One-time dump of text elements near the bottom of the page for debugging selectors."""
        try:
            result = await self._page.evaluate(
                """
                () => {
                    const viewH = window.innerHeight;
                    const elements = [];
                    const allDivs = document.querySelectorAll('div, span');
                    for (const el of allDivs) {
                        const rect = el.getBoundingClientRect();
                        if (rect.top < viewH * 0.60) continue;
                        if (rect.height > 100 || rect.height < 5) continue;
                        const text = el.textContent.trim();
                        if (!text || text.length < 2 || text.length > 100) continue;
                        // Only leaf elements (no div children with text)
                        let isLeaf = true;
                        for (const child of el.querySelectorAll('div, span')) {
                            if (child !== el && child.textContent.trim() === text) {
                                isLeaf = false;
                                break;
                            }
                        }
                        if (!isLeaf) continue;
                        const style = window.getComputedStyle(el);
                        elements.push({
                            tag: el.tagName,
                            text: text.substring(0, 60),
                            top: Math.round(rect.top),
                            height: Math.round(rect.height),
                            bg: style.backgroundColor,
                            role: el.getAttribute('role') || '',
                            ariaLabel: el.getAttribute('aria-label') || '',
                            className: (el.className || '').substring(0, 30),
                        });
                    }
                    return elements.slice(0, 20);
                }
                """
            )
            if result:
                print("🔍 DEBUG — Bottom-of-page elements:")
                for el in result:
                    print(f"   {el['tag']} top={el['top']} h={el['height']} bg={el['bg'][:30]} | \"{el['text']}\"")
        except Exception:
            pass

    async def _extract_captions(self) -> list[tuple[str, str]]:
        """
        Extract current caption text and speaker from the Google Meet DOM.

        Google Meet renders live captions in a specific container that appears
        at the bottom-center of the video area when someone speaks. The
        container only exists while captions are being displayed.

        Returns:
            List of (speaker, text) tuples for new caption content.
        """
        try:
            result = await self._page.evaluate(
                """
                () => {
                    const captions = [];

                    // Google Meet caption detection:
                    // When captions are on and someone speaks, Meet creates
                    // a caption overlay area. We detect it by looking for
                    // elements that:
                    //   1. Contain a person's name (speaker)
                    //   2. Followed by their spoken words
                    //   3. Positioned in the lower portion of the meeting area
                    //   4. NOT part of toolbar/buttons/tooltips/menus

                    // Known blocklist of UI text patterns to ignore
                    const ignorePatterns = [
                        /^turn (on|off)/i,
                        /^ctrl \\+/i,
                        /microphone|camera|captions|present|hand/i,
                        /[a-z]{3}-[a-z]{4}-[a-z]{3}/i,  // meeting code (anywhere in text)
                        /more options|leave call|chat with/i,
                        /^you$/i,
                        /^you\b/i,
                        /people|participants/i,
                        /activities|breakout/i,
                        /host controls/i,
                        /raise hand|lower hand/i,
                        /recording|transcript/i,
                        /screen sharing|presenting/i,
                        /joined|left the meeting/i,
                        /is presenting/i,
                        /muted/i,
                    ];

                    // Participant name overlay detection:
                    // Names on video tiles are typically ALL CAPS or Title Case,
                    // 1-3 words, no punctuation, no sentence structure.
                    function looksLikeParticipantName(text) {
                        const t = text.trim();
                        // All uppercase (e.g., "ADARSH BHARDWAJ")
                        if (t === t.toUpperCase() && t.length > 2 && /^[A-Z ]+$/.test(t)) return true;
                        // 1-3 words, no punctuation → likely a name label
                        const words = t.split(/[ ]+/);
                        if (words.length <= 3 && !/[.,!?;:]/.test(t) && t.length < 30) return true;
                        return false;
                    }

                    function isIgnoredText(text) {
                        const t = text.trim();
                        if (ignorePatterns.some(p => p.test(t))) return true;
                        if (looksLikeParticipantName(t)) return true;
                        return false;
                    }

                    function isToolbarElement(el) {
                        // Check if element or any ancestor is a toolbar/menu/dialog
                        let node = el;
                        while (node && node !== document.body) {
                            const role = node.getAttribute('role');
                            if (role === 'toolbar' || role === 'navigation' ||
                                role === 'menu' || role === 'menubar' ||
                                role === 'dialog' || role === 'listbox' ||
                                role === 'tooltip' || role === 'banner') {
                                return true;
                            }
                            // Check for aria-label patterns of toolbar buttons
                            const ariaLabel = (node.getAttribute('aria-label') || '').toLowerCase();
                            if (ariaLabel.includes('toolbar') ||
                                ariaLabel.includes('menu') ||
                                ariaLabel.includes('controls')) {
                                return true;
                            }
                            if (node.tagName === 'BUTTON' || node.tagName === 'INPUT' ||
                                node.tagName === 'SELECT' || node.tagName === 'TEXTAREA') {
                                return true;
                            }
                            node = node.parentElement;
                        }
                        return false;
                    }

                    // The caption area in Google Meet is rendered as a set of
                    // spans inside a container. The speaker name and their words
                    // are in adjacent spans/divs within this container.
                    //
                    // Strategy: Find all innermost text-containing divs in the
                    // bottom 35% of the screen that aren't UI elements.
                    // Caption divs typically have:
                    //   - A dark semi-transparent background
                    //   - White text
                    //   - Font size 14-20px
                    //   - Centered horizontally
                    //   - No interactive children

                    const viewH = window.innerHeight;
                    const viewW = window.innerWidth;

                    // Find caption-like containers
                    const allDivs = document.querySelectorAll('div');
                    const captionCandidates = [];

                    for (const div of allDivs) {
                        // Quick checks first
                        const text = div.textContent.trim();
                        if (!text || text.length < 3 || text.length > 150) continue;

                        // Skip if text matches known UI patterns
                        if (isIgnoredText(text)) continue;

                        const rect = div.getBoundingClientRect();

                        // Must be in the bottom 35% of viewport
                        if (rect.top < viewH * 0.60) continue;

                        // Reasonable size
                        if (rect.height > 80 || rect.height < 10) continue;
                        if (rect.width < 50 || rect.width > viewW * 0.9) continue;

                        // Must NOT be a toolbar/button/menu element
                        if (isToolbarElement(div)) continue;

                        // Must NOT have interactive children
                        if (div.querySelector('button, input, select, textarea, a, [role="button"]')) {
                            continue;
                        }

                        // Check if this is an innermost container (no child divs with text)
                        let hasChildDivWithText = false;
                        for (const child of div.querySelectorAll('div')) {
                            if (child !== div && child.textContent.trim().length > 2) {
                                hasChildDivWithText = true;
                                break;
                            }
                        }

                        // Prefer leaf-level containers but also consider parents
                        // if they provide speaker context
                        const style = window.getComputedStyle(div);
                        const bgColor = style.backgroundColor;
                        const color = style.color;

                        // Captions typically have dark/semi-transparent background
                        const hasDarkBg = bgColor.includes('rgba') ||
                                          bgColor.includes('0, 0, 0') ||
                                          bgColor.includes('33, 33, 33') ||
                                          bgColor.includes('32, 33, 36');

                        captionCandidates.push({
                            text: text,
                            rect: { top: rect.top, left: rect.left, width: rect.width, height: rect.height },
                            hasDarkBg: hasDarkBg,
                            isLeaf: !hasChildDivWithText,
                            el: div
                        });
                    }

                    // Score and filter candidates
                    for (const c of captionCandidates) {
                        // Only take candidates with dark background or leaf nodes
                        // near the very bottom
                        if (!c.hasDarkBg && c.rect.top < viewH * 0.75) continue;
                        if (!c.isLeaf) continue;

                        // Try to identify speaker name
                        let speaker = "Unknown";
                        let captionText = c.text;

                        // Look at parent for speaker name
                        const parent = c.el.parentElement;
                        if (parent) {
                            const siblings = Array.from(parent.children);
                            const myIndex = siblings.indexOf(c.el);

                            // Speaker name is often the first sibling
                            if (myIndex > 0) {
                                const prevSibling = siblings[myIndex - 1];
                                const prevText = prevSibling.textContent.trim();
                                if (prevText.length > 0 && prevText.length < 30 &&
                                    !isIgnoredText(prevText)) {
                                    speaker = prevText;
                                }
                            } else if (siblings.length >= 2) {
                                // Or first child of parent is name, rest is text
                                const firstText = siblings[0].textContent.trim();
                                if (firstText.length < 30 && firstText !== captionText &&
                                    !isIgnoredText(firstText)) {
                                    speaker = firstText;
                                    captionText = c.text.substring(firstText.length).trim();
                                    if (!captionText) captionText = c.text;
                                }
                            }
                        }

                        if (captionText && !isIgnoredText(captionText)) {
                            captions.push({
                                speaker: speaker,
                                text: captionText
                            });
                        }
                    }

                    // Deduplicate
                    const seen = new Set();
                    const unique = [];
                    for (const c of captions) {
                        const key = c.speaker + '|' + c.text;
                        if (!seen.has(key)) {
                            seen.add(key);
                            unique.push(c);
                        }
                    }

                    return unique;
                }
                """
            )

            if not result:
                return []

            new_captions = []
            for item in result:
                speaker = item.get("speaker", "Unknown")
                text = item.get("text", "").strip()
                if not text or len(text) < 3:
                    continue

                # Only emit if the text has changed
                if text != self._last_caption_text:
                    new_captions.append((speaker, text))
                    self._last_caption_speaker = speaker
                    self._last_caption_text = text

            return new_captions

        except Exception as e:
            logger.debug("caption_extract_error", error=str(e))
            return []

    # ── Emit transcript chunks ───────────────────────────────────────

    def _emit(self, speaker: str, text: str):
        """Convert a caption to a TranscriptChunk and emit it."""
        import re

        t = text.strip()

        # Skip meeting codes (exact or doubled)
        if self._meeting_code and self._meeting_code in t.lower():
            return
        if re.fullmatch(r'([a-z]{3}-[a-z]{4}-[a-z]{3})+', t, re.IGNORECASE):
            return

        # Skip ALL-CAPS names (video tile overlays like "ADARSH BHARDWAJ")
        if t == t.upper() and re.match(r'^[A-Z\s]+$', t) and len(t) > 2:
            return

        # Skip very short non-sentence text (likely UI labels)
        words = t.split()
        if len(words) <= 2 and not any(c in t for c in '.!?,;:'):
            return

        # Skip known UI phrases
        ui_phrases = [
            'turn on', 'turn off', 'ctrl +', 'more options',
            'leave call', 'raise hand', 'muted', 'presenting',
            'chat with', 'microphone', 'camera', 'captions',
            'participants', 'breakout', 'host controls',
        ]
        if any(phrase in t.lower() for phrase in ui_phrases):
            return

        print(f"📝 [{speaker}]: {text}")

        chunk = TranscriptChunk(
            text=text,
            timestamp=datetime.now(),
            speaker=speaker,
            is_final=True,
            confidence=0.95,  # Google's built-in captions are high quality
        )

        if self.on_transcript:
            self.on_transcript(chunk)
