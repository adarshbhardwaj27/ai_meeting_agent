"""
Microsoft 365 integration — API routes.

OAuth flow (browser-facing, no auth cookie required on callback):
  GET  /auth/microsoft/connect              → redirect to Microsoft consent
  GET  /auth/microsoft/callback             → exchange code, store, subscribe

Authenticated API (require_auth cookie):
  GET  /api/v1/microsoft/status             → connection status + filter config
  PUT  /api/v1/microsoft/filters            → update join filter rules
  DELETE /api/v1/microsoft/disconnect       → revoke tokens + delete subscription

Public webhook (called by Microsoft Graph, no auth):
  POST /webhooks/microsoft/calendar         → calendar change notifications
"""

import json
import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.session_store import store
from src.db.models import MicrosoftTenant
from src.db.session import get_db
from src.integrations.microsoft.encryption import decrypt, encrypt
from src.integrations.microsoft.graph_client import (
    delete_subscription,
    get_event,
    get_me,
    get_valid_token,
    renew_subscription,
    subscribe_calendar,
)
from src.integrations.microsoft.join_filters import should_join
from src.integrations.microsoft.oauth import (
    create_state_token,
    decode_state_token,
    exchange_code,
)

logger = structlog.get_logger()

_FRONTEND_URL = os.getenv("FRONTEND_URL", "https://taskora.in.net")
_NOTIFICATION_URL = os.getenv(
    "MICROSOFT_NOTIFICATION_URL",
    "https://api.taskora.in.net/webhooks/microsoft/calendar",
)

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class FiltersUpdate(BaseModel):
    join_mode: str = "all"
    # Mode-specific fields (all optional; validated by join_filters.should_join)
    organisers: Optional[list[str]] = None
    calendar_group: Optional[str] = None
    title_pattern: Optional[str] = None
    min_attendees: Optional[int] = None
    opt_in_address: Optional[str] = None


# ── OAuth flow ────────────────────────────────────────────────────────────────

@router.get("/auth/microsoft/connect")
async def microsoft_connect(request: Request):
    """Redirect the signed-in admin to the Microsoft OAuth consent screen."""
    from src.api.app import require_auth  # local import avoids circular dep
    auth = await require_auth(request)

    state = create_state_token(auth["company_id"])
    from src.integrations.microsoft.oauth import get_auth_url
    return RedirectResponse(get_auth_url(state))


@router.get("/auth/microsoft/callback")
async def microsoft_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Exchange the auth code for tokens, store them, and create a Graph subscription."""
    if error:
        logger.warning("microsoft_oauth_error", error=error, detail=error_description)
        return RedirectResponse(f"{_FRONTEND_URL}/dashboard?m365=error&reason={error}")

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter.")

    company_id = decode_state_token(state)
    if not company_id:
        raise HTTPException(status_code=400, detail="Invalid or expired state token.")

    # Exchange auth code for tokens
    result = await exchange_code(code)
    if not result:
        return RedirectResponse(f"{_FRONTEND_URL}/dashboard?m365=error&reason=token_exchange_failed")

    access_token = result["access_token"]
    refresh_token = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 3600)
    # Extract the customer's tenant ID from the ID token claims
    tenant_id = result.get("id_token_claims", {}).get("tid", "common")

    # Verify the token works and get user info
    try:
        me = await get_me(access_token)
    except Exception as e:
        logger.error("microsoft_me_failed", error=str(e))
        return RedirectResponse(f"{_FRONTEND_URL}/dashboard?m365=error&reason=graph_access_failed")

    now = datetime.now(timezone.utc)
    client_state = secrets.token_hex(16)

    # Upsert MicrosoftTenant row
    result_db = await db.execute(
        select(MicrosoftTenant).where(MicrosoftTenant.company_id == company_id)
    )
    tenant = result_db.scalar_one_or_none()

    # Delete old subscription if reconnecting
    if tenant and tenant.subscription_id:
        try:
            old_token = decrypt(tenant.access_token_enc)
            await delete_subscription(old_token, tenant.subscription_id)
        except Exception:
            pass  # Best-effort cleanup

    if tenant is None:
        tenant = MicrosoftTenant(company_id=company_id, connected_by=me.get("id", ""))
        db.add(tenant)

    tenant.tenant_id = tenant_id
    tenant.access_token_enc = encrypt(access_token)
    tenant.refresh_token_enc = encrypt(refresh_token) if refresh_token else ""
    tenant.token_expires_at = now + timedelta(seconds=expires_in)
    tenant.client_state = client_state
    tenant.connected_by = me.get("id", "")
    tenant.connected_at = now
    await db.flush()

    # Subscribe to calendar events
    try:
        sub = await subscribe_calendar(access_token, _NOTIFICATION_URL, client_state)
        tenant.subscription_id = sub["id"]
        expiry_str = sub.get("expirationDateTime", "")
        if expiry_str:
            tenant.subscription_expires_at = datetime.fromisoformat(
                expiry_str.replace("Z", "+00:00")
            )
    except Exception as e:
        logger.error("microsoft_subscribe_failed", error=str(e))
        # Don't abort — tenant is stored, user can reconnect

    await db.commit()
    logger.info("microsoft_connected", company_id=company_id, tenant_id=tenant_id)
    return RedirectResponse(f"{_FRONTEND_URL}/dashboard?m365=connected")


# ── Authenticated API ─────────────────────────────────────────────────────────

@router.get("/api/v1/microsoft/status")
async def microsoft_status(request: Request, db: AsyncSession = Depends(get_db)):
    from src.api.app import require_auth
    auth = await require_auth(request)

    result = await db.execute(
        select(MicrosoftTenant).where(MicrosoftTenant.company_id == auth["company_id"])
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        return {"connected": False}

    try:
        filters = json.loads(tenant.join_filters)
    except json.JSONDecodeError:
        filters = {"join_mode": "all"}

    return {
        "connected": True,
        "tenant_id": tenant.tenant_id,
        "subscription_active": tenant.subscription_id is not None,
        "subscription_expires_at": (
            tenant.subscription_expires_at.isoformat()
            if tenant.subscription_expires_at
            else None
        ),
        "connected_at": tenant.connected_at.isoformat(),
        "join_filters": filters,
    }


@router.put("/api/v1/microsoft/filters")
async def update_filters(
    body: FiltersUpdate, request: Request, db: AsyncSession = Depends(get_db)
):
    from src.api.app import require_auth
    auth = await require_auth(request)

    result = await db.execute(
        select(MicrosoftTenant).where(MicrosoftTenant.company_id == auth["company_id"])
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Microsoft 365 not connected.")

    filters = body.model_dump(exclude_none=True)
    tenant.join_filters = json.dumps(filters)
    await db.commit()
    return {"join_filters": filters}


@router.delete("/api/v1/microsoft/disconnect")
async def microsoft_disconnect(request: Request, db: AsyncSession = Depends(get_db)):
    from src.api.app import require_auth
    auth = await require_auth(request)

    result = await db.execute(
        select(MicrosoftTenant).where(MicrosoftTenant.company_id == auth["company_id"])
    )
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Microsoft 365 not connected.")

    # Best-effort: delete Graph subscription
    if tenant.subscription_id and tenant.access_token_enc:
        try:
            token = await get_valid_token(tenant, db)
            await delete_subscription(token, tenant.subscription_id)
        except Exception as e:
            logger.warning("microsoft_unsubscribe_failed", error=str(e))

    await db.delete(tenant)
    await db.commit()
    logger.info("microsoft_disconnected", company_id=auth["company_id"])
    return {"disconnected": True}


# ── Public webhook (Microsoft Graph calls this) ───────────────────────────────

@router.post("/webhooks/microsoft/calendar")
async def calendar_webhook(
    request: Request,
    validationToken: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Receives Microsoft Graph calendar change notifications.
    On first subscription Graph sends a GET/POST with validationToken — must
    echo it back as text/plain within 10 seconds.
    """
    # Subscription validation handshake
    if validationToken:
        return PlainTextResponse(validationToken)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    for notification in payload.get("value", []):
        subscription_id = notification.get("subscriptionId")
        client_state = notification.get("clientState")

        # Validate client_state to authenticate the notification
        result = await db.execute(
            select(MicrosoftTenant).where(
                MicrosoftTenant.subscription_id == subscription_id,
                MicrosoftTenant.client_state == client_state,
            )
        )
        tenant = result.scalar_one_or_none()
        if not tenant:
            logger.warning("graph_notification_unknown_subscription", sub_id=subscription_id)
            continue

        # The notification tells us which event changed
        resource_data = notification.get("resourceData", {})
        event_id = resource_data.get("id")
        if not event_id:
            continue

        try:
            token = await get_valid_token(tenant, db)
            event = await get_event(token, event_id)
        except Exception as e:
            logger.error("graph_event_fetch_failed", error=str(e))
            continue

        try:
            filters = json.loads(tenant.join_filters)
        except json.JSONDecodeError:
            filters = {"join_mode": "all"}

        join, teams_url = should_join(event, filters)
        if not join or not teams_url:
            continue

        # Don't join if already tracking this meeting
        existing = [s for s in store.all(company_id=tenant.company_id) if s.meeting_url == teams_url]
        if existing:
            continue

        try:
            store.create(meeting_url=teams_url, company_id=tenant.company_id)
            logger.info(
                "auto_join_scheduled",
                company_id=tenant.company_id,
                event_subject=event.get("subject"),
                teams_url=teams_url,
            )
        except Exception as e:
            logger.error("auto_join_failed", error=str(e))

    # Graph expects 202 Accepted
    from fastapi.responses import Response
    return Response(status_code=202)
