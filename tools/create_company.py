#!/usr/bin/env python3
"""
Provision a new company + admin user in the MeetMind database.

Usage (from repo root, with venv active):
    python tools/create_company.py \
        --name  "Acme Corp" \
        --email admin@acme.com \
        --password <secret>

To reset a user's password:
    python tools/create_company.py \
        --name  "Acme Corp" \
        --email admin@acme.com \
        --password <new_secret> \
        --update-password
"""

import argparse
import asyncio
import os
import re
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from src.auth.service import hash_password
from src.db.models import Company, User
from src.db.session import AsyncSessionLocal, init_db


async def run(name: str, email: str, password: str, update_password: bool) -> None:
    await init_db()

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    async with AsyncSessionLocal() as db:
        # Check if company with this slug already exists
        result = await db.execute(select(Company).where(Company.slug == slug))
        company = result.scalar_one_or_none()

        if not company:
            company = Company(name=name, slug=slug)
            db.add(company)
            await db.flush()
            print(f"  Company '{name}' created  (id={company.id})")
        else:
            print(f"  Company '{name}' already exists  (id={company.id})")

        # Check if user already exists
        result = await db.execute(select(User).where(User.email == email.lower()))
        user = result.scalar_one_or_none()

        if not user:
            user = User(
                company_id=company.id,
                email=email.lower(),
                password_hash=hash_password(password),
            )
            db.add(user)
            print(f"  User '{email}' created  (id={user.id})")
        elif update_password:
            user.password_hash = hash_password(password)
            print(f"  Password updated for '{email}'")
        else:
            print(f"  User '{email}' already exists — pass --update-password to reset")

        await db.commit()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision a MeetMind company + admin user")
    parser.add_argument("--name", required=True, help="Company display name")
    parser.add_argument("--email", required=True, help="Admin user email")
    parser.add_argument("--password", required=True, help="Admin user password")
    parser.add_argument(
        "--update-password",
        action="store_true",
        help="Reset password if user already exists",
    )
    args = parser.parse_args()
    asyncio.run(run(args.name, args.email, args.password, args.update_password))
