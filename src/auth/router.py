"""
Auth router — /auth/login, /auth/logout, /auth/me
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.service import authenticate_user, create_access_token, decode_access_token
from src.db.models import Company, User
from src.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

_COOKIE = "access_token"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


class LoginRequest(BaseModel):
    email: str
    password: str


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=_COOKIE,
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
        path="/",
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/login")
async def login(
    body: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    user = await authenticate_user(db, body.email, body.password)
    if not user:
        # Constant-time response — don't reveal whether email exists
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token(user.company_id, user.id)
    _set_auth_cookie(response, token)

    result = await db.execute(
        select(Company).where(Company.id == user.company_id)
    )
    company = result.scalar_one_or_none()

    return {
        "user": {"id": user.id, "email": user.email},
        "company": {"id": company.id, "name": company.name} if company else None,
    }


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(
        key=_COOKIE, path="/", httponly=True, secure=True, samesite="lax"
    )
    return {"ok": True}


@router.get("/me")
async def me(request: Request, db: AsyncSession = Depends(get_db)):
    token = request.cookies.get(_COOKIE)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated."
        )

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
        )

    result = await db.execute(
        select(User, Company)
        .join(Company, User.company_id == Company.id)
        .where(User.id == payload["user_id"])
    )
    row = result.first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found."
        )

    user, company = row
    return {
        "user": {"id": user.id, "email": user.email},
        "company": {"id": company.id, "name": company.name},
    }
