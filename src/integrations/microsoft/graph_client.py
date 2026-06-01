"""Async Microsoft Graph API client."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import structlog

from src.integrations.microsoft.encryption import decrypt, encrypt
from src.integrations.microsoft.oauth import refresh_access_token

logger = structlog.get_logger()
_GRAPH = "https://graph.microsoft.com/v1.0"


async def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}


# ── Token management ──────────────────────────────────────────────────────────

async def get_valid_token(tenant, db) -> str:
    """Return a fresh access token, refreshing from DB if the stored one is expired."""
    now = datetime.now(timezone.utc)
    expires = tenant.token_expires_at

    if expires and expires.replace(tzinfo=timezone.utc) > now + timedelta(minutes=5):
        return decrypt(tenant.access_token_enc)

    # Refresh needed
    rt = decrypt(tenant.refresh_token_enc)
    result = await refresh_access_token(tenant.tenant_id, rt)
    if not result:
        raise RuntimeError("Microsoft token refresh failed — tenant must reconnect.")

    tenant.access_token_enc = encrypt(result["access_token"])
    if "refresh_token" in result:
        tenant.refresh_token_enc = encrypt(result["refresh_token"])
    tenant.token_expires_at = now + timedelta(seconds=result.get("expires_in", 3600))
    await db.commit()
    return result["access_token"]


# ── Graph API calls ───────────────────────────────────────────────────────────

async def get_me(access_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{_GRAPH}/me", headers=await _headers(access_token))
        resp.raise_for_status()
        return resp.json()


async def get_event(access_token: str, event_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_GRAPH}/me/events/{event_id}",
            headers=await _headers(access_token),
        )
        resp.raise_for_status()
        return resp.json()


async def subscribe_calendar(
    access_token: str, notification_url: str, client_state: str
) -> dict:
    """Create a Graph subscription for /me/events. Expires in 72 h (Graph max for calendar)."""
    expiry = (datetime.now(timezone.utc) + timedelta(hours=72)).strftime(
        "%Y-%m-%dT%H:%M:%S.0000000Z"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_GRAPH}/subscriptions",
            headers=await _headers(access_token),
            json={
                "changeType": "created,updated",
                "notificationUrl": notification_url,
                "resource": "/me/events",
                "expirationDateTime": expiry,
                "clientState": client_state,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def renew_subscription(access_token: str, subscription_id: str) -> dict:
    """Extend a Graph subscription by another 72 hours."""
    expiry = (datetime.now(timezone.utc) + timedelta(hours=72)).strftime(
        "%Y-%m-%dT%H:%M:%S.0000000Z"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"{_GRAPH}/subscriptions/{subscription_id}",
            headers=await _headers(access_token),
            json={"expirationDateTime": expiry},
        )
        resp.raise_for_status()
        return resp.json()


async def delete_subscription(access_token: str, subscription_id: str) -> None:
    """Delete a Graph subscription (called on disconnect)."""
    async with httpx.AsyncClient() as client:
        await client.delete(
            f"{_GRAPH}/subscriptions/{subscription_id}",
            headers=await _headers(access_token),
        )
