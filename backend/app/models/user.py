# models/user.py
#
# WHY: Every layer of the app that touches user data needs a shared
# definition of what a "user" looks like. This is that definition.
# FastAPI uses it to serialize API responses. dependencies.py uses it
# to type the current_user object injected into endpoints.

from pydantic import BaseModel
from datetime import datetime
from typing import Literal


# ── What a user profile looks like coming OUT of the database ──────────────
# This is the shape of a row from the `profiles` table.
# Used as the return type of get_current_user() in dependencies.py.

class UserProfile(BaseModel):
    id:          str
    github_id:   str
    username:    str
    email:       str | None
    avatar_url:  str | None
    plan:        Literal["free", "pro"] = "free"
    repos_used:  int = 0
    created_at:  datetime | None = None

    class Config:
        # Allows constructing from a dict with extra keys (e.g. raw DB row)
        # Extra fields are silently ignored instead of raising an error
        extra = "ignore"


# ── What we send to the frontend in /auth/me ──────────────────────────────
# Subset of UserProfile — we don't expose everything to the client.

class UserResponse(BaseModel):
    id:         str
    username:   str
    email:      str | None
    avatar_url: str | None
    plan:       Literal["free", "pro"]
    repos_used: int
