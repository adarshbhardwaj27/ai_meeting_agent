"""Evaluate meeting join filter rules against a Microsoft Graph calendar event."""

import re
from typing import Optional


def extract_teams_url(event: dict) -> Optional[str]:
    """Pull a Teams join URL from a Graph event, checking multiple locations."""
    # 1. onlineMeeting.joinUrl (most reliable)
    online = event.get("onlineMeeting") or {}
    if online.get("joinUrl"):
        return online["joinUrl"]

    # 2. Scan the HTML body
    body = (event.get("body") or {}).get("content", "")
    m = re.search(r"https://teams\.microsoft\.com/l/meetup-join/[^\s\"<>']+", body)
    if m:
        return m.group(0)

    # 3. Location displayName
    location = (event.get("location") or {}).get("displayName", "")
    m = re.search(r"https://teams\.microsoft\.com/[^\s\"<>']+", location)
    if m:
        return m.group(0)

    return None


def should_join(event: dict, filters: dict) -> tuple[bool, Optional[str]]:
    """
    Check whether the bot should join this calendar event.

    Returns:
        (True, teams_url) if the bot should join.
        (False, None)     otherwise.

    Filter modes
    ------------
    all            — join every Teams meeting (default)
    organiser_list — only if organiser email is in filters["organisers"]
    calendar_group — only if event has a category matching filters["calendar_group"]
    title_regex    — only if subject matches filters["title_pattern"] (case-insensitive)
    min_attendees  — only if attendee count >= filters["min_attendees"]
    opt_in         — only if filters["opt_in_address"] is among the attendees
    """
    teams_url = extract_teams_url(event)
    if not teams_url:
        return False, None

    if event.get("isCancelled", False):
        return False, None

    mode = filters.get("join_mode", "all")

    if mode == "all":
        return True, teams_url

    if mode == "organiser_list":
        organiser = (
            event.get("organizer", {})
            .get("emailAddress", {})
            .get("address", "")
            .lower()
        )
        allowed = {e.lower() for e in filters.get("organisers", [])}
        return organiser in allowed, teams_url

    if mode == "calendar_group":
        target = filters.get("calendar_group", "").lower()
        categories = [c.lower() for c in event.get("categories", [])]
        return target in categories, teams_url

    if mode == "title_regex":
        pattern = filters.get("title_pattern", "")
        subject = event.get("subject", "")
        try:
            return bool(re.search(pattern, subject, re.IGNORECASE)), teams_url
        except re.error:
            return False, teams_url

    if mode == "min_attendees":
        count = len(event.get("attendees", []))
        minimum = int(filters.get("min_attendees", 2))
        return count >= minimum, teams_url

    if mode == "opt_in":
        opt_in_addr = filters.get("opt_in_address", "alex@meetmind.ai").lower()
        addresses = {
            a.get("emailAddress", {}).get("address", "").lower()
            for a in event.get("attendees", [])
        }
        return opt_in_addr in addresses, teams_url

    return False, teams_url
