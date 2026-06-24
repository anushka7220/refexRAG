#You don't want to create a new client on every request — that's wasteful and slow. because FastAPI app will make dozens of database calls 
#This script creates one client at startup and makes it available everywhere.
#Same singleton pattern as config.py, but for the database connection.
#Your API endpoints that return user data should use anon + the user's JWT so RLS protects them automatically.

#@lru_cache on the admin client but not the user client — the admin client is always the same regardless of who's asking, so caching it makes sense. The user client changes per request because the JWT is different for every user. Caching it would mean user A gets user B's data. Never cache per-request state.

#RLS is your second security layer, not your first — your FastAPI middleware checks the JWT first. If that passes, the user client with RLS is your second check at the database level. Even if someone bypassed your FastAPI auth somehow, they'd hit the RLS wall and see nothing. Two independent layers.
from supabase import create_client, Client
from functools import lru_cache
from app.core.config import settings

#admin client
# BYPASSES row-level security entirely.
# Use this ONLY for:
#   - Ingestion worker writing chunks, decision nodes, contributor data
#   - Webhook handlers updating repo status
#   - Background jobs that run as the system, not as a user
#
# NEVER use this in endpoints that return data to users —
# it would expose every user's data regardless of who's asking.
@lru_cache
def get_admin_client() -> Client:
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_ROLE_KEY,
    )

# Single instance imported by services that need system-level access
supabase_admin: Client = get_admin_client()

#user client
# RESPECTS row-level security.
# Pass the user's JWT (extracted from the Authorization header) and Supabase
# will automatically enforce RLS — the client can only see rows where
# the policy allows that user's auth.uid().
#
# Use this in:
#   - Chat endpoints (user sees only their own sessions)
#   - Repo listing (user sees only their linked repos)
#   - Any endpoint that returns user-specific data
#
# WHY a factory function and not a singleton:
# Each user has a different JWT. We need a fresh client configured
# with that specific token per request. This is not cached.
def get_user_client(jwt: str) -> Client:
    client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_ANON_KEY,
    )
    # Inject the user's JWT so Supabase knows who is making the request.
    # This is what makes auth.uid() work inside RLS policies.
    client.auth.set_session(access_token=jwt, refresh_token="")
    return client

#The execute() helper — Supabase-py doesn't raise exceptions on query failures by default.
#It returns a response object with an error field you have to check manually. Without this helper, you'd forget to check response.
#error somewhere and silently swallow a database failure. Wrapping it once here means every query either returns data or raises clearly.

# Supabase-py returns a response object, not a direct result.
# Every query you run looks like:
#   response = supabase_admin.table("chunks").insert({...}).execute()
#
# If something goes wrong, the error is buried in response.
# This helper extracts the data or raises a clear Python exception.

def execute(response) -> list[dict]:
    if hasattr(response, "error") and response.error:
        raise ValueError(f"Supabase query failed: {response.error.message}")
    return response.data




