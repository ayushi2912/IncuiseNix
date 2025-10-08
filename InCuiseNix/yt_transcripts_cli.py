#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YouTube transcripts CLI with playlist and batch support.

Default behavior: fetch available captions only (no downloads). If captions are missing
and you pass --allow-fallback, it will download media (audio-only if --audio-only) and
transcribe locally with faster-whisper.

Examples:
  # Playlist: captions only (skip videos without captions)
  python yt_transcripts_cli.py "https://www.youtube.com/playlist?list=PL123..." --srt --verbose

  # Single video: captions only
  python yt_transcripts_cli.py "https://youtu.be/dQw4w9WgXcQ" --srt

  # Batch file of URLs/IDs: captions only
  python yt_transcripts_cli.py --batch-file urls.txt --srt --language en

  # Allow local transcription fallback when captions are missing (requires ffmpeg)
  python yt_transcripts_cli.py "https://youtu.be/dQw4w9WgXcQ" --allow-fallback --audio-only --model-size small --srt --verbose
"""

from __future__ import annotations

import argparse
import sys
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def parse_video_id(url: str) -> Optional[str]:
    if YOUTUBE_ID_RE.match(url):
        return url
    try:
        u = urlparse(url)
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
        m = re.search(r"/embed/([A-Za-z0-9_-]{11})", path)
        if m:
            return m.group(1)
        m = re.search(r"/shorts/([A-Za-z0-9_-]{11})", path)
        if m:
            return m.group(1)
    return None


def try_fetch_transcript_via_api(video_id: str, languages: List[str]) -> Optional[List[dict]]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except Exception:
        return None


def segments_to_plain_text(segments: List[dict]) -> str:
    lines: List[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines) + ("\n" if lines else "")


def format_timestamp_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: List[dict]) -> str:
    blocks: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        dur = float(seg.get("duration", 0.0))
        end = start + max(dur, 0.0)
        text = (seg.get("text") or "").strip()
        blocks.append(f"{idx}\n{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n{text}\n")
    return "\n".join(blocks)


def get_video_metadata(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        import yt_dlp
    except ImportError:
        return None, parse_video_id(url)
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if isinstance(info, dict) and (info.get('_type') == 'playlist' or 'entries' in info):
                return None, None
            title = info.get("title") if isinstance(info, dict) else None
            vid = info.get("id") if isinstance(info, dict) else None
            return title, vid
    except Exception:
        return None, parse_video_id(url)


def iterate_playlist_entries(url: str) -> List[dict]:
    try:
        import yt_dlp
    except ImportError:
        return []
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": True, "noplaylist": False}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            entries = []
            if isinstance(info, dict) and (info.get('_type') == 'playlist' or 'entries' in info):
                for e in info.get('entries') or []:
                    if not isinstance(e, dict):
                        continue
                    vid = e.get('id') or parse_video_id(e.get('url') or '')
                    title = e.get('title') or vid
                    if vid:
                        entries.append({'id': vid, 'title': title, 'url': f"https://www.youtube.com/watch?v={vid}"})
            return entries
    except Exception:
        return []


def download_media(url: str, out_dir: Path, audio_only: bool = False) -> Path:
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Please install it: pip install yt-dlp")
    ensure_dir(out_dir)
    output_tmpl = str(out_dir / "%(title).200B [%(id)s].%(ext)s")
    if audio_only:
        ydl_opts = {
            "outtmpl": output_tmpl,
            "format": "bestaudio/best",
            "postprocessors": [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
                'preferredquality': '5',
            }],
            "restrictfilenames": False,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
    else:
        ydl_opts = {
            "outtmpl": output_tmpl,
            "format": "bestvideo*+bestaudio/best",
            "merge_output_format": "mp4",
            "restrictfilenames": False,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = None
        if isinstance(info, dict):
            rds = info.get("requested_downloads") or []
            if rds and isinstance(rds, list):
                fp = rds[0].get("filepath")
                if fp:
                    filepath = Path(fp)
            if not filepath:
                vid = info.get("id")
                title = info.get("title")
                ext = info.get("ext", "mp4" if not audio_only else "m4a")
                if title and vid:
                    filepath = out_dir / f"{title} [{vid}].{ext}"
        if not filepath:
            raise RuntimeError("Failed to determine downloaded file path. Update yt-dlp and retry.")
        return filepath


def transcribe_with_faster_whisper(media_path: Path, model_size: str = "small", language: Optional[str] = None,
                                   beam_size: int = 5) -> Tuple[List[dict], dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper is not installed. Please install it: pip install faster-whisper")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    seg_iter, info = model.transcribe(
        str(media_path),
        beam_size=beam_size,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    segments: List[dict] = []
    for seg in seg_iter:
        segments.append({
            "text": seg.text.strip(),
            "start": float(seg.start) if seg.start is not None else 0.0,
            "end": float(seg.end) if seg.end is not None else 0.0,
        })
    for s in segments:
        s["duration"] = max(0.0, s["end"] - s["start"])
    return segments, {"language": getattr(info, "language", None), "duration": getattr(info, "duration", None)}


def process_single(url_or_id: str, out_dir: Path, language: Optional[str], model_size: str, srt: bool,
                   allow_fallback: bool, audio_only: bool, verbose: bool) -> int:
    title, vid = get_video_metadata(url_or_id)
    if not vid:
        vid = parse_video_id(url_or_id)
    if not vid:
        if verbose:
            print(f"Skip: Could not parse video ID from input: {url_or_id}")
        return 2
    if not title:
        title = vid
    base_name = f"{title} [{vid}]"
    txt_path = out_dir / f"{base_name}.transcript.txt"
    srt_path = out_dir / f"{base_name}.transcript.srt"
    if verbose:
        print(f"Processing: {title} ({vid})")
    lang_prefs: List[str] = []
    if language:
        lang_prefs.append(language)
    lang_prefs.extend(["en", "en-US", "en-GB"])
    if verbose:
        print("  - Trying YouTubeTranscriptApi...")
    transcript_segments = try_fetch_transcript_via_api(vid, lang_prefs)
    if transcript_segments:
        plain = segments_to_plain_text(transcript_segments)
        txt_path.write_text(plain, encoding="utf-8")
        if srt:
            srt_txt = segments_to_srt(transcript_segments)
            srt_path.write_text(srt_txt, encoding="utf-8")
        print(f"Transcript saved: {txt_path}")
        if srt:
            print(f"SRT saved: {srt_path}")
        if verbose:
            print("  - Source: YouTube captions (no download)")
        return 0
    if not allow_fallback:
        if verbose:
            print("  - No captions found. Skipping (fallback disabled).")
        return 0
    if not has_ffmpeg():
        print("Error: FFmpeg not found on PATH. Please install FFmpeg and try again.", file=sys.stderr)
        return 3
    if verbose:
        print(f"  - Downloading {'audio only' if audio_only else 'video'} for transcription...")
    try:
        media_path = download_media(url_or_id, out_dir, audio_only=audio_only)
    except Exception as e:
        print(f"Error downloading media: {e}", file=sys.stderr)
        return 4
    if verbose:
        print(f"  - Downloaded: {media_path}")
        print("  - Transcribing with faster-whisper...")
    try:
        segments, info = transcribe_with_faster_whisper(media_path, model_size=model_size, language=language)
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return 5
    plain = segments_to_plain_text(segments)
    txt_path.write_text(plain, encoding="utf-8")
    if srt:
        srt_txt = segments_to_srt(segments)
        srt_path.write_text(srt_txt, encoding="utf-8")
    print(f"Transcript saved: {txt_path}")
    if srt:
        print(f"SRT saved: {srt_path}")
    print(f"Media saved: {media_path}")
    if verbose and info:
        lang = info.get("language")
        dur = info.get("duration")
        if lang:
            print(f"  - Detected language: {lang}")
        if dur:
            print(f"  - Duration (s): {dur}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch transcripts for YouTube videos/playlists; optional local fallback.")
    parser.add_argument("url", nargs="?", help="YouTube video/playlist URL or 11-char video ID")
    parser.add_argument("--batch-file", help="Path to a text file with URLs/IDs (one per line)")
    parser.add_argument("--out-dir", default="outputs", help="Directory to store transcripts (and downloaded media if fallback is used)")
    parser.add_argument("--model-size", default="small", help="faster-whisper model size: tiny, base, small, medium, large-v2, etc.")
    parser.add_argument("--language", default=None, help="Language code hint for transcription (e.g., en, hi, fr). Leave empty for auto-detect.")
    parser.add_argument("--srt", action="store_true", help="Also save SRT subtitle file.")
    parser.add_argument("--allow-fallback", action="store_true", help="If no captions, download and transcribe locally.")
    parser.add_argument("--audio-only", action="store_true", help="When fallback is allowed, download only audio for transcription.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")

    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    inputs: List[str] = []
    if args.batch_file:
        p = Path(args.batch_file)
        if not p.exists():
            print(f"Error: batch file not found: {p}", file=sys.stderr)
            return 2
        inputs.extend([line.strip() for line in p.read_text(encoding='utf-8').splitlines() if line.strip()])
    if args.url:
        inputs.append(args.url)
    if not inputs:
        parser.print_help(sys.stderr)
        return 2

    overall_rc = 0
    for inp in inputs:
        entries = iterate_playlist_entries(inp)
        if entries:
            if args.verbose:
                print(f"Found playlist with {len(entries)} entries. Processing...")
            for e in entries:
                rc = process_single(e['url'], out_dir, args.language, args.model_size, args.srt,
                                    args.allow_fallback, args.audio_only, args.verbose)
                if rc != 0 and overall_rc == 0:
                    overall_rc = rc
        else:
            rc = process_single(inp, out_dir, args.language, args.model_size, args.srt,
                                args.allow_fallback, args.audio_only, args.verbose)
            if rc != 0 and overall_rc == 0:
                overall_rc = rc
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
