"""Authenticated HUB availability without exposing environment values."""

from fastapi import APIRouter, Depends

from parcel_app.core.security import current_user

from .credentials import configured

router = APIRouter()


@router.get("/sources/hub/status")
def status(user: dict = Depends(current_user)):
    """Return only whether the server has its HUB service configuration."""
    return {"configured": configured()}
