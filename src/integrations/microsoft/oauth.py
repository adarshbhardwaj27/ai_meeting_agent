"""Microsoft OAuth 2.0 helpers — auth URL, token exchange, state JWT."""

import asyncio
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import msal
from jose import jwt as jose_jwt

_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
_REDIRECT_URI = os.getenv("MICROSOFT_REDIRECT_URI", "")
_JWT_SECRET = os.getenv("JWT_SECRET", "dev")
_AUTHORITY = "https://login.microsoftonline.com/common"
_SCOPES = [
    "https://graph.microsoft.com/Calendars.Read",
    "https://graph.microsoft.com/User.Read",
    "offline_access",
]


def _msal_app(tenant_id: str = "common") -> msal.ConfidentialClientApplication:
    return msal.ConfidentialClientApplication(
        client_id=_CLIENT_ID,
        client_credential=_CLIENT_SECRET,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
    )


# ── State token (CSRF protection + company_id carrier) ───────────────────────

def create_state_token(company_id: str) -> str:
    payload = {
        "company_id": company_id,
        "nonce": secrets.token_hex(8),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
    }
    return jose_jwt.encode(payload, _JWT_SECRET, algorithm="HS256")


def decode_state_token(state: str) -> Optional[str]:
    try:
        payload = jose_jwt.decode(state, _JWT_SECRET, algorithms=["HS256"])
        return payload.get("company_id")
    except Exception:
        return None


# ── OAuth flow ────────────────────────────────────────────────────────────────

def get_auth_url(state: str) -> str:
    """Build the Microsoft OAuth consent URL."""
    app = _msal_app()
    return app.get_authorization_request_url(
        scopes=_SCOPES,
        redirect_uri=_REDIRECT_URI,
        state=state,
        prompt="select_account",
    )


async def exchange_code(code: str) -> Optional[dict]:
    """Exchange an auth code for tokens. Returns the MSAL result dict or None."""
    app = _msal_app()
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: app.acquire_token_by_authorization_code(
            code=code,
            scopes=_SCOPES,
            redirect_uri=_REDIRECT_URI,
        ),
    )
    if "error" in result:
        return None
    return result


async def refresh_access_token(tenant_id: str, refresh_token: str) -> Optional[dict]:
    """Use a refresh_token to get a new access_token."""
    app = _msal_app(tenant_id)
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: app.acquire_token_by_refresh_token(
            refresh_token=refresh_token,
            scopes=_SCOPES,
        ),
    )
    if "error" in result:
        return None
    return result
