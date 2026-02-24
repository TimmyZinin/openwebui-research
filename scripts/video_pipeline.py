#!/usr/bin/env python3
"""
AI Video Factory — Video Pipeline v2
Generates 60-second vertical videos with avatar + b-roll cutaways.

Flow:
1. Script generation (via LLM) → segments: avatar vs b-roll
2. TTS audio (ElevenLabs)
3. Avatar video segments (MiniMax S2V-01)
4. B-roll images (Pexels/Pixabay/Picsum)
5. Assembly: interleave avatar + b-roll + audio → final MP4
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

# Blogger configurations with avatar settings
BLOGGER_PROFILES = {
    "tim": {
        "name": "Tim",
        "reference_image": "https://files.catbox.moe/46ivt1.jpg",
        "voice": "Adam",  # ElevenLabs male voice
        "tone": "уверенный эксперт, бизнес-ментор",
        "topics_lang": "Russian",
    },
    "lisa": {
        "name": "Lisa",
        "reference_image": None,  # Will use local file path
        "voice": "Laura",
        "tone": "тёплая старшая подруга, карьерный ментор",
        "topics_lang": "Russian",
    },
}


def generate_script(topic: str, blogger_name: str = "Tim", mode: str = "avatar_broll") -> dict:
    """Generate video script via LLM with avatar/b-roll segment markers.

    Args:
        topic: Video topic.
        blogger_name: Name of the blogger.
        mode: "avatar_broll" (interleaved) or "broll_only" (old style).

    Returns:
        dict with title, segments (avatar/broll), hashtags, etc.
    """
    import requests
    import re

    profile = BLOGGER_PROFILES.get(blogger_name.lower(), {})
    tone = profile.get("tone", "friendly expert")

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

    if mode == "avatar_broll":
        system_prompt = f"""You are a script writer for {blogger_name}'s video blog.
Write a 60-second video script in {profile.get('topics_lang', 'Russian')} about the given topic.

The video alternates between AVATAR segments (person talking to camera) and B-ROLL cutaway segments (illustrative footage).

Output JSON:
{{
  "title": "catchy title in Russian",
  "hook": "first 3 seconds hook text",
  "segments": [
    {{"type": "avatar", "text": "narration text for this segment (~15 words)", "prompt": "person speaking to camera confidently [Static shot]", "duration": 6}},
    {{"type": "broll", "text": "narration continues over b-roll (~15 words)", "prompt": "office workspace with laptop and coffee, warm lighting", "duration": 6}},
    {{"type": "avatar", "text": "more narration (~15 words)", "prompt": "person gestures while explaining a concept [Static shot]", "duration": 6}},
    {{"type": "broll", "text": "narration over b-roll (~15 words)", "prompt": "team meeting in modern coworking space", "duration": 6}},
    {{"type": "avatar", "text": "final call to action (~15 words)", "prompt": "person smiles and points at camera [Static shot]", "duration": 6}}
  ],
  "full_script": "complete narration text for TTS (all segments combined)",
  "hashtags": ["#tag1", "#tag2", "#tag3"]
}}

Rules:
- Tone: {tone}
- Alternate avatar and b-roll segments (start and end with avatar)
- 5 segments total (3 avatar + 2 b-roll), each ~6 seconds = 30 seconds avatar + 12 seconds b-roll
- Avatar prompts: describe the person's action/emotion for video generation (English)
- B-roll prompts: describe the scene in English, photorealistic style
- full_script: the complete narration for TTS (Russian), flowing naturally
- End with call-to-action"""
    else:
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
- Tone: {tone}
- Practical advice
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
                "max_tokens": 2000,
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
                # Parse JSON from response
                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
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


def generate_tts(text: str, output_path: Path, voice: str = "Laura") -> Path:
    """Generate TTS audio via ElevenLabs."""
    import requests

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    voices = {
        "Laura": "FGY2WhTYpPnrIDTdsKH5",
        "Rachel": "21m00Tcm4TlvDq8ikWAM",
        "Adam": "pNInz6obpgDQGcFmaJgB",
        "Antoni": "ErXwobaYiN019PkySvjV",
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


def generate_avatar_segments(
    segments: list[dict],
    reference_image_url: str,
    output_dir: Path,
) -> list[Path]:
    """Generate avatar video segments via MiniMax S2V-01.

    Args:
        segments: List of segment dicts with type="avatar" and "prompt" key.
        reference_image_url: Public URL to face reference photo.
        output_dir: Directory to save MP4 segments.

    Returns:
        List of paths to avatar video segments (in order).
    """
    from minimax_api import MiniMaxClient

    client = MiniMaxClient()
    avatar_segments = [s for s in segments if s["type"] == "avatar"]
    results = []

    # Submit all tasks first (parallel generation)
    tasks = []
    for i, seg in enumerate(avatar_segments):
        try:
            task_id = client.generate_avatar_video(
                prompt=seg["prompt"],
                reference_image_url=reference_image_url,
            )
            tasks.append((i, task_id))
            time.sleep(2)
        except Exception as e:
            print(f"[Avatar] Segment {i} submission failed: {e}")
            tasks.append((i, None))

    # Download all completed videos
    for i, task_id in tasks:
        if not task_id:
            results.append(None)
            continue
        try:
            output_path = output_dir / f"avatar_{i:02d}.mp4"
            client.wait_and_download(task_id, output_path)
            results.append(output_path)
        except Exception as e:
            print(f"[Avatar] Segment {i} failed: {e}")
            results.append(None)

    return results


def generate_broll(prompts: list[str], output_dir: Path) -> list[Path]:
    """Fetch b-roll images from multiple free sources."""
    import requests
    import urllib.parse

    images = []
    pexels_key = os.getenv("PEXELS_API_KEY", "")

    for i, prompt in enumerate(prompts):
        img_saved = False
        try:
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

            if not img_saved:
                img_resp = requests.get(
                    "https://picsum.photos/1080/1920",
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


def broll_image_to_video(image_path: Path, duration: float, output_path: Path) -> Path:
    """Convert a b-roll image to a Ken Burns video segment."""
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(duration),
        "-i", str(image_path),
        "-vf", (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            f"zoompan=z='min(zoom+0.001,1.15)':d={int(duration*25)}:s=1080x1920:fps=25"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"B-roll video failed: {result.stderr[-300:]}")
    return output_path


def assemble_interleaved(
    video_segments: list[Path],
    audio_path: Path,
    output_path: Path,
    music_path: Path | None = None,
) -> Path:
    """Concatenate video segments and overlay audio.

    Args:
        video_segments: Ordered list of MP4 segments (avatar + b-roll videos).
        audio_path: Full narration audio (MP3).
        output_path: Final output MP4.
        music_path: Optional background music.
    """
    if not video_segments:
        raise RuntimeError("No video segments to assemble")

    # Create concat list file
    concat_file = output_path.parent / "concat_list.txt"
    with open(concat_file, "w") as f:
        for seg in video_segments:
            f.write(f"file '{seg}'\n")

    # Concat all video segments
    concat_video = output_path.parent / "concat_raw.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-r", "25",
        str(concat_video),
    ]
    print("[Assembly] Concatenating video segments...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Concat failed: {result.stderr[-500:]}")

    # Add audio to concatenated video
    cmd = ["ffmpeg", "-y", "-i", str(concat_video), "-i", str(audio_path)]

    if music_path and music_path.exists():
        cmd.extend(["-i", str(music_path)])
        filter_audio = (
            "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[voice];"
            "[2:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            "volume=0.15[music];"
            "[voice][music]amix=inputs=2:duration=shortest[outa]"
        )
        cmd.extend([
            "-filter_complex", filter_audio,
            "-map", "0:v", "-map", "[outa]",
        ])
    else:
        cmd.extend(["-map", "0:v", "-map", "1:a"])

    cmd.extend([
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ])

    print("[Assembly] Adding audio track...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Audio overlay failed: {result.stderr[-500:]}")

    # Cleanup temp files
    concat_file.unlink(missing_ok=True)
    concat_video.unlink(missing_ok=True)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Assembly] Done: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def assemble_broll_only(
    audio_path: Path,
    broll_images: list[Path],
    output_path: Path,
    music_path: Path | None = None,
) -> Path:
    """Legacy: assemble b-roll slideshow + audio (no avatar)."""
    if not broll_images:
        raise RuntimeError("No b-roll images to assemble")

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    segment_duration = duration / len(broll_images)

    inputs = []
    filter_parts = []

    for i, img in enumerate(broll_images):
        inputs.extend(["-loop", "1", "-t", str(segment_duration), "-i", str(img)])
        filter_parts.append(
            f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,zoompan=z='min(zoom+0.001,1.15)':"
            f"d={int(segment_duration*25)}:s=1080x1920:fps=25[v{i}]"
        )

    concat_inputs = "".join(f"[v{i}]" for i in range(len(broll_images)))
    filter_parts.append(f"{concat_inputs}concat=n={len(broll_images)}:v=1:a=0[outv]")
    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y", *inputs, "-i", str(audio_path)]

    if music_path and music_path.exists():
        cmd.extend(["-i", str(music_path)])
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
        "-map", "[outv]", *audio_map,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ])

    print("[Assembly] Running FFmpeg (b-roll only)...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Assembly] Done: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def produce_video(topic: str, blogger_name: str = "Tim", mode: str = "auto") -> Path:
    """Full pipeline: topic → final MP4.

    Args:
        topic: Video topic.
        blogger_name: "Tim", "Lisa", etc.
        mode: "auto" (avatar if MiniMax available, else b-roll), "avatar", "broll".
    """
    profile = BLOGGER_PROFILES.get(blogger_name.lower(), BLOGGER_PROFILES["tim"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = OUTPUT_DIR / f"{blogger_name.lower()}_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine mode
    has_minimax = bool(os.getenv("MINIMAX_API_KEY"))
    has_reference = bool(profile.get("reference_image"))

    if mode == "auto":
        mode = "avatar" if (has_minimax and has_reference) else "broll"

    use_avatar = mode == "avatar"

    print(f"\n{'='*50}")
    print(f"AI Video Factory — {profile['name']}")
    print(f"Topic: {topic}")
    print(f"Mode: {'Avatar + B-roll' if use_avatar else 'B-roll only'}")
    print(f"Output: {work_dir}")
    print(f"{'='*50}\n")

    # 1. Generate script
    script_mode = "avatar_broll" if use_avatar else "broll_only"
    print(f"[1/5] Generating script ({script_mode})...")
    script = generate_script(topic, profile["name"], script_mode)
    (work_dir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2))

    # 2. Generate TTS
    print("[2/5] Generating TTS audio...")
    tts_text = script.get("full_script") or script.get("script", "")
    if not tts_text and "segments" in script:
        tts_text = " ".join(s["text"] for s in script["segments"])
    audio_path = generate_tts(tts_text, work_dir / "voice.mp3", profile["voice"])

    if use_avatar and "segments" in script:
        # 3. Generate avatar segments via MiniMax
        print("[3/5] Generating avatar segments (MiniMax S2V-01)...")
        avatar_segments = generate_avatar_segments(
            script["segments"],
            profile["reference_image"],
            work_dir,
        )

        # 4. Generate b-roll segments
        print("[4/5] Generating b-roll segments...")
        broll_prompts = [s["prompt"] for s in script["segments"] if s["type"] == "broll"]
        broll_images = generate_broll(broll_prompts, work_dir)

        # Convert b-roll images to video segments
        broll_videos = []
        for i, img in enumerate(broll_images):
            broll_seg = script["segments"]
            broll_duration = 6  # default
            for s in broll_seg:
                if s["type"] == "broll":
                    broll_duration = s.get("duration", 6)
                    break
            vid_path = work_dir / f"broll_vid_{i:02d}.mp4"
            broll_image_to_video(img, broll_duration, vid_path)
            broll_videos.append(vid_path)

        # 5. Interleave: avatar, broll, avatar, broll, avatar
        print("[5/5] Assembling interleaved video...")
        video_segments = []
        avatar_idx, broll_idx = 0, 0
        for seg in script["segments"]:
            if seg["type"] == "avatar":
                if avatar_idx < len(avatar_segments) and avatar_segments[avatar_idx]:
                    video_segments.append(avatar_segments[avatar_idx])
                avatar_idx += 1
            elif seg["type"] == "broll":
                if broll_idx < len(broll_videos):
                    video_segments.append(broll_videos[broll_idx])
                broll_idx += 1

        if not video_segments:
            print("[WARNING] No video segments generated. Falling back to b-roll only.")
            # Fallback to b-roll only
            all_broll = generate_broll(
                [s["prompt"] for s in script["segments"]],
                work_dir,
            )
            output_path = assemble_broll_only(
                audio_path, all_broll,
                work_dir / f"{blogger_name.lower()}_final.mp4",
            )
        else:
            music_path = ASSETS_DIR / "music" / "default_bg.mp3"
            output_path = assemble_interleaved(
                video_segments, audio_path,
                work_dir / f"{blogger_name.lower()}_final.mp4",
                music_path if music_path.exists() else None,
            )
    else:
        # B-roll only mode (legacy)
        print("[3/4] Generating b-roll images...")
        broll_prompts = script.get("broll_prompts", [])
        if not broll_prompts and "segments" in script:
            broll_prompts = [s["prompt"] for s in script["segments"]]
        broll_images = generate_broll(broll_prompts, work_dir)

        if not broll_images:
            print("[WARNING] No b-roll generated. Cannot produce video.")
            return None

        print("[4/4] Assembling video (b-roll only)...")
        music_path = ASSETS_DIR / "music" / "default_bg.mp3"
        output_path = assemble_broll_only(
            audio_path, broll_images,
            work_dir / f"{blogger_name.lower()}_final.mp4",
            music_path if music_path.exists() else None,
        )

    # Save metadata
    meta = {
        "blogger": profile["name"],
        "topic": topic,
        "title": script.get("title", topic),
        "hashtags": script.get("hashtags", []),
        "mode": mode,
        "output": str(output_path),
        "created": datetime.now().isoformat(),
    }
    (work_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\n[DONE] Video ready: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Factory Pipeline")
    parser.add_argument("topic", nargs="?", default="Как AI меняет рынок труда в 2026")
    parser.add_argument("--blogger", default="Tim", help="Blogger name (Tim, Lisa)")
    parser.add_argument("--mode", default="auto", choices=["auto", "avatar", "broll"],
                        help="Video mode: auto, avatar (MiniMax), broll (images only)")
    args = parser.parse_args()

    produce_video(args.topic, args.blogger, args.mode)
