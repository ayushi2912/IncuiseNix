from __future__ import annotations

from typing import Optional, List
import re
from urllib.parse import urlparse, parse_qs

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _parse_video_id(url_or_id: str) -> Optional[str]:
    if YOUTUBE_ID_RE.match(url_or_id):
        return url_or_id
    try:
        u = urlparse(url_or_id)
    except Exception:
        return None
    host = (u.netloc or "").lower()
    path = u.path or ""
    if "youtu.be" in host:
        vid = path.lstrip("/")
        return vid if YOUTUBE_ID_RE.match(vid) else None
    if "youtube.com" in host or "m.youtube.com" in host or "www.youtube.com" in host:
        qs = parse_qs(u.query)
        if "v" in qs:
            vid = qs["v"][0]
            return vid if YOUTUBE_ID_RE.match(vid) else None
    return None


def get_transcript(video_url_or_id: str, languages: Optional[List[str]] = None) -> Optional[str]:
    """
    Fetch YouTube transcript text if available. Returns a joined plain-text string,
    or None if no captions are available.

    This function intentionally avoids downloading any media. It only uses the
    YouTube captions API via youtube-transcript-api.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        # Dependency not installed; let caller handle None
        return None

    vid = _parse_video_id(video_url_or_id)
    if not vid:
        return None

    lang_prefs = languages[:] if languages else []
    if not lang_prefs:
        # Common English variants as defaults
        lang_prefs = ["en", "en-US", "en-GB"]

    try:
        segments = YouTubeTranscriptApi.get_transcript(vid, languages=lang_prefs)
        # segments: [{"text": str, "start": float, "duration": float}, ...]
        parts = []
        for seg in segments:
            t = (seg.get("text") or "").strip()
            if t:
                parts.append(t)
        return "\n".join(parts) if parts else ""
    except Exception:
        # No transcript available or other API error
        return None
