"""
Small Redis-backed cache helpers.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import redis

from app.core.config import Settings, get_settings


@lru_cache()
def get_redis_client() -> redis.Redis | None:
    """Return a shared Redis client when configured."""
    settings = get_settings()
    if not settings.redis_url:
        return None
    try:
        return redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
    except Exception:
        return None


def build_cache_key(prefix: str, value: str) -> str:
    """Build a compact stable cache key."""
    digest = hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def cache_get_json(key: str) -> Any | None:
    """Read JSON data from Redis."""
    client = get_redis_client()
    if client is None:
        return None
    try:
        payload = client.get(key)
        if not payload:
            return None
        return json.loads(payload)
    except Exception:
        return None


def cache_set_json(key: str, value: Any, ttl_seconds: int) -> None:
    """Write JSON data to Redis."""
    client = get_redis_client()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=True))
    except Exception:
        return
