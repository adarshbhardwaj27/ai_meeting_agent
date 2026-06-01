"""
SQLAlchemy ORM models — Company, User, MicrosoftTenant.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    users: Mapped[list["User"]] = relationship("User", back_populates="company")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    company_id: Mapped[str] = mapped_column(
        String, ForeignKey("companies.id"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    company: Mapped["Company"] = relationship("Company", back_populates="users")


class MicrosoftTenant(Base):
    """Per-company Microsoft 365 connection — tokens, subscription, and join filters."""

    __tablename__ = "microsoft_tenants"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    company_id: Mapped[str] = mapped_column(
        String, ForeignKey("companies.id"), unique=True, nullable=False
    )
    # The customer's AAD tenant ID (from the id_token after OAuth)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    # Fernet-encrypted tokens
    access_token_enc: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    refresh_token_enc: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Graph calendar subscription
    subscription_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    subscription_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Random secret sent with every Graph notification — used to validate callbacks
    client_state: Mapped[str] = mapped_column(String(100), nullable=False)
    # JSON: { "join_mode": "all" | "organiser_list" | "calendar_group" |
    #                           "title_regex" | "min_attendees" | "opt_in",
    #         ...mode-specific keys... }
    join_filters: Mapped[str] = mapped_column(String, default='{"join_mode":"all"}')
    connected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    connected_by: Mapped[str] = mapped_column(String(100), nullable=False)  # user_id

    company: Mapped["Company"] = relationship("Company")
