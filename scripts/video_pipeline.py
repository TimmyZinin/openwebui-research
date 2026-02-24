#!/usr/bin/env python3
"""
AI Video Factory — Video Pipeline
Generates 60-second vertical videos for AI bloggers.

Flow:
1. Script generation (via LLM)
2. TTS audio (ElevenLabs or Qwen3-TTS)
3. B-roll images (Gemini Flash via OpenRouter, free 500/day)
4. Avatar video (HeyGen via browser automation)
5. Assembly: avatar + b-roll + music → final MP4
6. Upload to social networks via RUVDS
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path("/opt/ai-video")
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
SCRIPTS_DIR = BASE_DIR / "scripts"
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"

# Load env
from dotenv import load_dotenv
load_dotenv(CONFIG_DIR / ".env")


def generate_script(topic: str, blogger_name: str = "Lisa") -> dict:
    """Generate video script via LLM (DeepSeek/GLM/Groq)."""
    import requests

    # Try providers in order: GLM via OpenRouter (free) → Qwen → Groq
    providers = [
        {
            "name": "GLM-4.5 (OpenRouter)",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key": os.getenv("OPENROUTER_API_KEY"),
            "model": "z-ai/glm-4.5-air:free",
        },
        {
            "name": "Qwen3 (OpenRouter)",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key": os.getenv("OPENROUTER_API_KEY"),
            "model": "qwen/qwen3-next-80b-a3b-instruct:free",
        },
        {
            "name": "Groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": os.getenv("GROQ_API_KEY"),
            "model": "llama-3.3-70b-versatile",
        },
    ]

    system_prompt = f"""You are a script writer for {blogger_name}'s video blog.
Write a 60-second video script in Russian about the given topic.

Output JSON:
{{
  "title": "catchy title",
  "hook": "first 3 seconds hook",
  "script": "full narration text (60 seconds ~150 words)",
  "broll_prompts": ["scene 1 description", "scene 2", "scene 3", "scene 4"],
  "hashtags": ["#tag1", "#tag2", "#tag3"]
}}

Rules:
- Warm, friendly tone ("старшая подруга")
- Practical career advice
- End with call-to-action
- B-roll prompts in English, photorealistic style"""

    for provider in providers:
        if not provider["key"]:
            continue
        try:
            payload = {
                "model": provider["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Topic: {topic}"},
                ],
                "temperature": 0.7,
                "max_tokens": 1500,
            }
            resp = requests.post(
                provider["url"],
                headers={
                    "Authorization": f"Bearer {provider['key']}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Parse JSON from response (may be wrapped in ```json...```)
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    # Try finding outermost braces
                    start = content.find("{")
                    end = content.rfind("}")
                    if start >= 0 and end > start:
                        content = content[start:end + 1]
                print(f"[Script] Generated via {provider['name']}")
                return json.loads(content)
            elif resp.status_code == 429:
                print(f"[Script] {provider['name']} rate limited, trying next...")
                continue
        except Exception as e:
            print(f"[Script] {provider['name']} failed: {e}")
            continue

    raise RuntimeError("All LLM providers failed")


def generate_broll(prompts: list[str], output_dir: Path) -> list[Path]:
    """Fetch b-roll images from multiple free sources."""
    import requests
    import urllib.parse

    images = []
    pexels_key = os.getenv("PEXELS_API_KEY", "")

    for i, prompt in enumerate(prompts):
        img_saved = False
        try:
            # Source 1: Pexels API (free, 200 req/hour, needs API key)
            if pexels_key and not img_saved:
                resp = requests.get(
                    "https://api.pexels.com/v1/search",
                    headers={"Authorization": pexels_key},
                    params={"query": prompt, "per_page": 1, "orientation": "portrait"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    photos = resp.json().get("photos", [])
                    if photos:
                        img_url = photos[0]["src"]["large2x"]
                        img_resp = requests.get(img_url, timeout=60)
                        if img_resp.status_code == 200:
                            img_path = output_dir / f"broll_{i:02d}.jpg"
                            img_path.write_bytes(img_resp.content)
                            images.append(img_path)
                            print(f"[B-roll] Pexels: {img_path.name}")
                            img_saved = True

            # Source 2: Pixabay API (free, no key needed for limited use)
            if not img_saved:
                pixabay_key = os.getenv("PIXABAY_API_KEY", "47108394-fa9a0da5bbd7b157bb39b7d38")
                query = urllib.parse.quote(prompt[:100])
                resp = requests.get(
                    f"https://pixabay.com/api/?key={pixabay_key}&q={query}&image_type=photo&orientation=vertical&per_page=3",
                    timeout=30,
                )
                if resp.status_code == 200:
                    hits = resp.json().get("hits", [])
                    if hits:
                        img_url = hits[0].get("largeImageURL", hits[0].get("webformatURL", ""))
                        if img_url:
                            img_resp = requests.get(img_url, timeout=60)
                            if img_resp.status_code == 200:
                                img_path = output_dir / f"broll_{i:02d}.jpg"
                                img_path.write_bytes(img_resp.content)
                                images.append(img_path)
                                print(f"[B-roll] Pixabay: {img_path.name}")
                                img_saved = True

            # Source 3: Lorem Picsum (random high-quality photos)
            if not img_saved:
                img_resp = requests.get(
                    f"https://picsum.photos/1080/1920",
                    timeout=30, allow_redirects=True,
                )
                if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                    img_path = output_dir / f"broll_{i:02d}.jpg"
                    img_path.write_bytes(img_resp.content)
                    images.append(img_path)
                    print(f"[B-roll] Picsum: {img_path.name} (random)")
                    img_saved = True

            if not img_saved:
                print(f"[B-roll] Image {i}: all sources failed")

        except Exception as e:
            print(f"[B-roll] Image {i} failed: {e}")

        time.sleep(1)

    return images


def generate_tts(text: str, output_path: Path, voice: str = "Laura") -> Path:
    """Generate TTS audio via ElevenLabs."""
    import requests

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    # Voice IDs (ElevenLabs)
    voices = {
        "Laura": "FGY2WhTYpPnrIDTdsKH5",  # Warm female Russian
        "Rachel": "21m00Tcm4TlvDq8ikWAM",
    }

    voice_id = voices.get(voice, voices["Laura"])

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        },
        timeout=120,
    )

    if resp.status_code == 200:
        output_path.write_bytes(resp.content)
        print(f"[TTS] Generated: {output_path.name}")
        return output_path
    else:
        raise RuntimeError(f"ElevenLabs failed: {resp.status_code} {resp.text[:200]}")


def assemble_video(
    audio_path: Path,
    broll_images: list[Path],
    output_path: Path,
    music_path: Path | None = None,
) -> Path:
    """Assemble final video: b-roll slideshow + audio + optional music."""

    if not broll_images:
        raise RuntimeError("No b-roll images to assemble")

    # Get audio duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    segment_duration = duration / len(broll_images)

    # Build FFmpeg filter for Ken Burns effect slideshow
    inputs = []
    filter_parts = []

    for i, img in enumerate(broll_images):
        inputs.extend(["-loop", "1", "-t", str(segment_duration), "-i", str(img)])
        # Ken Burns: slow zoom in
        filter_parts.append(
            f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,zoompan=z='min(zoom+0.001,1.15)':"
            f"d={int(segment_duration*25)}:s=1080x1920:fps=25[v{i}]"
        )

    # Concat all video segments
    concat_inputs = "".join(f"[v{i}]" for i in range(len(broll_images)))
    filter_parts.append(f"{concat_inputs}concat=n={len(broll_images)}:v=1:a=0[outv]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-i", str(audio_path),
    ]

    if music_path and music_path.exists():
        cmd.extend(["-i", str(music_path)])
        # Mix voice + music (music at 15% volume)
        audio_idx = len(broll_images) + 1
        filter_complex += (
            f";[{len(broll_images)}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[voice];"
            f"[{audio_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"volume=0.15[music];"
            f"[voice][music]amix=inputs=2:duration=shortest[outa]"
        )
        audio_map = ["-map", "[outa]"]
    else:
        audio_map = ["-map", f"{len(broll_images)}:a"]

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        *audio_map,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ])

    print(f"[Assembly] Running FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Assembly] Done: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def produce_video(topic: str, blogger_name: str = "Lisa") -> Path:
    """Full pipeline: topic → final MP4."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = OUTPUT_DIR / f"{blogger_name.lower()}_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"AI Video Factory — {blogger_name}")
    print(f"Topic: {topic}")
    print(f"Output: {work_dir}")
    print(f"{'='*50}\n")

    # 1. Generate script
    print("[1/4] Generating script...")
    script = generate_script(topic, blogger_name)
    (work_dir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2))

    # 2. Generate TTS
    print("[2/4] Generating TTS audio...")
    audio_path = generate_tts(script["script"], work_dir / "voice.mp3")

    # 3. Generate b-roll
    print("[3/4] Generating b-roll images...")
    broll_images = generate_broll(script["broll_prompts"], work_dir)

    if not broll_images:
        print("[WARNING] No b-roll generated. Cannot produce video.")
        return None

    # 4. Assemble video
    print("[4/4] Assembling video...")
    music_path = ASSETS_DIR / "music" / "default_bg.mp3"
    output_path = assemble_video(
        audio_path, broll_images,
        work_dir / f"{blogger_name.lower()}_final.mp4",
        music_path if music_path.exists() else None,
    )

    # Save metadata
    meta = {
        "blogger": blogger_name,
        "topic": topic,
        "title": script["title"],
        "hashtags": script["hashtags"],
        "output": str(output_path),
        "created": datetime.now().isoformat(),
    }
    (work_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\n[DONE] Video ready: {output_path}")
    return output_path


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "5 ошибок на собеседовании, которые стоят вам оффер"
    produce_video(topic)
