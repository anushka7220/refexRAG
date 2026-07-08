#three endpoints are there that handles github oauth login.
#1. redirects the user to the github's consent screen
#2. that github redirects bacl to with a code 
#3. returns current users file

#when user clicks "login with github" this is the file which sends them to github
#Your server exchanges that code for a GitHub token, fetches their profile, creates or updates their row in profiles, issues your own JWT, and redirects back to the frontend with the token in the URL.
#frontend stores the token and sends it with every subsequent request.

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.security import (
    exchange_github_code,
    fetch_github_user,
    create_access_token,
)
from app.core.supabase import supabase_admin, execute
from app.core.dependencies import get_current_user
from app.models.user import UserResponse
from fastapi import Depends
from app.models.user import UserProfile

import structlog

log = structlog.get_logger(__name__)

router = APIRouter()

#step1. redirect to github
@router.get("/github")
async def login_with_github():
    """ 
    The scopes we request:
        read:user   — username, avatar, profile info
        user:email  — primary email address
        public_repo — read access to public repos for ingestion
    """
    github_oauth_url = (
        "https://github.com/login/oauth/authorize"
        f"?client_id={settings.GITHUB_CLIENT_ID}"
        f"&scope=read:user,user:email,public_repo"
        f"&redirect_uri={settings.BACKEND_URL}/auth/callback"
    )
    return RedirectResponse(url=github_oauth_url)

#step 2- handle github callback
@router.get("/callback")
async def github_callback(code: str):
    """
    GitHub redirects here after the user approves the OAuth screen.
 
    Args:
        code: Short-lived single-use code from GitHub's redirect query param.
 
    Redirects to the frontend with the JWT in the URL fragment.
    The frontend reads it from window.location.hash and stores it.
 
    WHY URL FRAGMENT (#) NOT QUERY PARAM (?):
    URL fragments are not sent to servers in HTTP requests, so the token
    cannot leak in server logs. Query params are visible to any server
    the redirect passes through.
    """
    try:
        #exchange code for github access token
        github_token = await exchange_github_code(code)
    except ValueError as e:
        log.error("github_oauth_error", error=str(e))
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/auth/error?reason=github_exchange_failed"
        )
    
    try:
        #fetch hithub user profile using that token
        github_user = await fetch_github_user(github_token)
    except Exception as e:
        log.error("github_profile_fetch_failed", error=str(e))
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/auth/error?reason=github_profile_failed"
        )
    
    try:
        #upsert usrr into our profile table
        # If the user exists, update their username and avatar in case they changed.
        # If they do not exist, create a new row.
        user_id = upsert_profile(github_user)
    except Exception as e:
        log.error("profile_upsert_failed", error=str(e))
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/auth/error?reason=profile_upsert_failed"
        )

    #issue on our own JWT cntainign the user's supabase UUI
    jwt = create_access_token(user_id)

    log.info("login_suceess", user_id=user_id, username=github_user["username"])

    #redirect to frontend with token in URL fragment
    return RedirectResponse(url=f"{settings.FRONTEND_URL}/auth/success#{jwt}")

#step 3; return current user
@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserProfile = Depends(get_current_user)):
    """
    Returns the current logged-in user's profile.
 
    The frontend calls this on page load to check if the user is logged in.
    If the JWT is valid, it returns the user's profile. If not, it returns 401.
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        avatar_url=current_user.avatar_url,
        plan=current_user.plan,
        repos_used=current_user.repos_used,
    )

@router.post("/logout")
async def logout():
    """
    Logout endpoint. Since we use stateless JWTs, there is no server
    side session to invalidate. The frontend is responsible for deleting
    the stored token.
 
    This endpoint exists so the frontend has a clean place to call on
    logout and we can add server side token blacklisting later if needed.
    """
    return {"message": "logged out"}
 
#nternal helper
def _upsert_profile(github_user: dict) -> str:
    """
    Creates or updates a user profile in Supabase.
 
    Uses github_id as the stable identifier since GitHub usernames
    can change but IDs never do.
 
    Args:
        github_user: Dict from fetch_github_user() with keys:
                     github_id, username, avatar_url, email.
 
    Returns:
        The Supabase UUID of the user's profile row.
    """
    #checking if user already exists by github_id
    response = (
        supabase_admin
        .table("profiles")
        .select("id")
        .eq("github_id", github_user["github_id"])
        .execute()
    )
    rows = execute(response)

    if rows:
        #user exists, update their username and avatar in case they changed
        user_id = rows[0]["id"]
        supabase_admin.table("profiles").update({
            "username": github_user["username"],
            "avatar_url": github_user["avatar_url"],
            "email": github_user["email"],
        }).eq("id", user_id).execute()

        log.info("profile_updated", user_id=user_id)
        return user_id
    
    else:
        #new user, create profile row
        response = supabase_admin.table("profiles").insert({
            "github_id":  github_user["github_id"],
            "username":   github_user["username"],
            "avatar_url": github_user["avatar_url"],
            "email":      github_user.get("email"),
            "plan":       "free",
            "repos_used": 0,
        }).execute()
 
        rows = execute(response)
        user_id = rows[0]["id"]
 
        log.info("profile_created", user_id=user_id, username=github_user["username"])
        return user_id
 

