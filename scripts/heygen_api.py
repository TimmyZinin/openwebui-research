#!/usr/bin/env python3
"""
HeyGen API Client — Photo Avatar video generation.

Endpoints:
- POST /v2/photo_avatar/photo/generate — create photo avatar
- POST /v2/video/generate — create video with avatar
- GET /v1/video_status.get?video_id=XXX — check video status

Pricing (pay-as-you-go):
- Photo Avatar III: $1/min
- Photo Avatar IV: $6/min
- Photo Avatar creation: $1-4/call
"""

import os
import time
import json
import requests
from pathlib import Path

BASE_URL = "https://api.heygen.com"


def get_api_key() -> str:
    key = os.getenv("HEYGEN_API_KEY")
    if not key:
        raise RuntimeError("HEYGEN_API_KEY not set")
    return key


def headers() -> dict:
    return {
        "X-Api-Key": get_api_key(),
        "Content-Type": "application/json",
    }


def upload_photo(photo_path: str) -> str:
    """Upload a photo to HeyGen and return asset_id."""
    key = get_api_key()
    with open(photo_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/v1/asset",
            headers={"X-Api-Key": key},
            files={"file": (Path(photo_path).name, f, "image/png")},
        )
    resp.raise_for_status()
    data = resp.json()
    asset_id = data.get("data", {}).get("asset_id") or data.get("data", {}).get("id")
    print(f"[HeyGen] Uploaded photo: {asset_id}")
    return asset_id


def create_photo_avatar(photo_asset_id: str) -> str:
    """Create a Photo Avatar from an uploaded photo."""
    resp = requests.post(
        f"{BASE_URL}/v2/photo_avatar/photo/generate",
        headers=headers(),
        json={"photo_asset_id": photo_asset_id},
    )
    resp.raise_for_status()
    data = resp.json()
    avatar_id = data.get("data", {}).get("talking_photo_id")
    print(f"[HeyGen] Created photo avatar: {avatar_id}")
    return avatar_id


def create_video(
    talking_photo_id: str,
    script_text: str,
    voice_id: str = "en-US-JennyNeural",  # Default Azure voice
    background_color: str = "#FFFFFF",
) -> str:
    """Create a video with a photo avatar speaking the script."""
    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "talking_photo",
                    "talking_photo_id": talking_photo_id,
                    "talking_style": "expressive",
                    "expression": "default",
                    "super_resolution": True,
                },
                "voice": {
                    "type": "text",
                    "voice_id": voice_id,
                    "input_text": script_text,
                    "speed": 1.0,
                },
                "background": {
                    "type": "color",
                    "value": background_color,
                },
            }
        ],
        "dimension": {"width": 1080, "height": 1920},  # Vertical 9:16
    }

    resp = requests.post(
        f"{BASE_URL}/v2/video/generate",
        headers=headers(),
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    video_id = data.get("data", {}).get("video_id")
    print(f"[HeyGen] Video generation started: {video_id}")
    return video_id


def wait_for_video(video_id: str, timeout: int = 600, poll_interval: int = 15) -> str:
    """Poll for video completion and return download URL."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(
            f"{BASE_URL}/v1/video_status.get",
            headers=headers(),
            params={"video_id": video_id},
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        status = data.get("status")

        if status == "completed":
            url = data.get("video_url")
            print(f"[HeyGen] Video ready: {url}")
            return url
        elif status == "failed":
            error = data.get("error", "Unknown error")
            raise RuntimeError(f"Video generation failed: {error}")
        else:
            print(f"[HeyGen] Status: {status} ({int(time.time() - start)}s elapsed)")
            time.sleep(poll_interval)

    raise TimeoutError(f"Video not ready after {timeout}s")


def download_video(url: str, output_path: str) -> str:
    """Download the generated video."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[HeyGen] Downloaded: {output_path} ({size_mb:.1f} MB)")
    return output_path


def list_voices() -> list:
    """List available TTS voices."""
    resp = requests.get(f"{BASE_URL}/v1/voice.list", headers=headers())
    resp.raise_for_status()
    voices = resp.json().get("data", {}).get("voices", [])
    return voices


def list_avatars() -> list:
    """List available avatars (including photo avatars)."""
    resp = requests.get(f"{BASE_URL}/v1/avatar.list", headers=headers())
    resp.raise_for_status()
    avatars = resp.json().get("data", {}).get("avatars", [])
    return avatars


if __name__ == "__main__":
    # Example: list available voices for Russian
    voices = list_voices()
    ru_voices = [v for v in voices if "ru" in v.get("language", "").lower()]
    print(f"\nRussian voices ({len(ru_voices)}):")
    for v in ru_voices[:10]:
        print(f"  {v.get('voice_id')}: {v.get('display_name')} ({v.get('gender')})")
