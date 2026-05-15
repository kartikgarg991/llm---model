from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from datetime import datetime, timedelta, timezone

import redis


SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "7200"))
ACTIVE_SESSIONS_KEY = "active_sessions"
CLEANUP_COOLDOWN_KEY = "cleanup:last_run"
IST = timezone(timedelta(hours=5, minutes=30))
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise ValueError("REDIS_URL is missing")
    _client = redis.from_url(redis_url, decode_responses=True, max_connections=50)
    return _client


def _session_key(session_id):
    return f"session:{session_id}"


def _chat_key(session_id):
    return f"chat:{session_id}"


def _normalize_history(history):
    if not isinstance(history, list):
        return []

    normalized = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        user = turn.get("user")
        assistant = turn.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            continue
        normalized.append({"user": user, "assistant": assistant})
    return normalized[-5:]


def touch_session(session_id):
    client = _get_client()
    now = int(time.time())
    expires_at = now + SESSION_TTL_SECONDS
    session_info = {
        "last_active_at": now,
        "last_active_at_ist": datetime.fromtimestamp(now, IST).strftime("%d-%m-%Y %H:%M:%S IST"),
        "expires_at": expires_at,
        "expires_at_ist": datetime.fromtimestamp(expires_at, IST).strftime("%d-%m-%Y %H:%M:%S IST"),
    }

    client.set(_session_key(session_id), json.dumps(session_info), ex=SESSION_TTL_SECONDS)
    if client.exists(_chat_key(session_id)):
        client.expire(_chat_key(session_id), SESSION_TTL_SECONDS)
    client.zadd(ACTIVE_SESSIONS_KEY, {session_id: expires_at})
    return expires_at


def get_history(session_id):
    client = _get_client()
    raw_history = client.get(_chat_key(session_id))
    if not raw_history:
        return []
    try:
        history = json.loads(raw_history)
    except json.JSONDecodeError:
        return []
    return _normalize_history(history)


def save_history(session_id, history):
    client = _get_client()
    trimmed_history = _normalize_history(history)
    client.set(_chat_key(session_id), json.dumps(trimmed_history), ex=SESSION_TTL_SECONDS)
    touch_session(session_id)
    return trimmed_history


def get_expired_sessions(limit=100):
    client = _get_client()
    now = int(time.time())
    return client.zrangebyscore(ACTIVE_SESSIONS_KEY, "-inf", now, start=0, num=limit)


def mark_cleanup_if_due(cooldown_seconds):
    client = _get_client()
    readable_time = datetime.now(IST).strftime("%d-%m-%Y %H:%M:%S IST")
    return bool(client.set(CLEANUP_COOLDOWN_KEY, readable_time, ex=cooldown_seconds, nx=True))


def clear_session(session_id):
    client = _get_client()
    client.delete(_session_key(session_id), _chat_key(session_id))
    client.zrem(ACTIVE_SESSIONS_KEY, session_id)
