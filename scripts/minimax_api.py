#!/usr/bin/env python3
"""
MiniMax (Hailuo) Video Generation API Client.

Supports:
- S2V-01: Subject reference (face-consistent avatar from 1 photo)
- T2V: Text-to-video
- I2V: Image-to-video

Usage:
    from minimax_api import MiniMaxClient
    client = MiniMaxClient(api_key="sk-...")
    task_id = client.generate_avatar_video("Tim talks about AI", "https://example.com/face.jpg")
    video_path = client.wait_and_download(task_id, Path("output.mp4"))
"""

import os
import time
import requests
from pathlib import Path


class MiniMaxClient:
    BASE_URL = "https://api.minimax.io/v1"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY not set")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def generate_avatar_video(
        self,
        prompt: str,
        reference_image_url: str,
        model: str = "S2V-01",
        prompt_optimizer: bool = False,
    ) -> str:
        """Generate avatar video using S2V-01 (subject reference).

        Args:
            prompt: Scene description (max 2000 chars). Use [Camera commands].
            reference_image_url: Public URL to face reference image.
            model: Model name (default S2V-01).
            prompt_optimizer: Auto-optimize prompt (default False for precise control).

        Returns:
            task_id for polling status.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "subject_reference": [
                {
                    "type": "character",
                    "image": [reference_image_url],
                }
            ],
            "prompt_optimizer": prompt_optimizer,
        }

        resp = self.session.post(
            f"{self.BASE_URL}/video_generation",
            json=payload,
            timeout=60,
        )
        data = resp.json()

        if data.get("base_resp", {}).get("status_code", -1) != 0:
            raise RuntimeError(
                f"MiniMax generation failed: {data.get('base_resp', {}).get('status_msg', resp.text[:300])}"
            )

        task_id = data["task_id"]
        print(f"[MiniMax] Task created: {task_id}")
        return task_id

    def generate_t2v(
        self,
        prompt: str,
        model: str = "MiniMax-Hailuo-02",
        duration: int = 6,
        resolution: str = "768P",
    ) -> str:
        """Generate text-to-video (no face reference).

        Returns:
            task_id for polling status.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "prompt_optimizer": True,
        }

        resp = self.session.post(
            f"{self.BASE_URL}/video_generation",
            json=payload,
            timeout=60,
        )
        data = resp.json()

        if data.get("base_resp", {}).get("status_code", -1) != 0:
            raise RuntimeError(
                f"MiniMax T2V failed: {data.get('base_resp', {}).get('status_msg', resp.text[:300])}"
            )

        task_id = data["task_id"]
        print(f"[MiniMax] T2V task: {task_id}")
        return task_id

    def generate_i2v(
        self,
        prompt: str,
        first_frame_url: str,
        model: str = "MiniMax-Hailuo-02",
        duration: int = 6,
        resolution: str = "768P",
    ) -> str:
        """Generate image-to-video (animate a still image).

        Returns:
            task_id for polling status.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "first_frame_image": first_frame_url,
            "duration": duration,
            "resolution": resolution,
            "prompt_optimizer": True,
        }

        resp = self.session.post(
            f"{self.BASE_URL}/video_generation",
            json=payload,
            timeout=60,
        )
        data = resp.json()

        if data.get("base_resp", {}).get("status_code", -1) != 0:
            raise RuntimeError(
                f"MiniMax I2V failed: {data.get('base_resp', {}).get('status_msg', resp.text[:300])}"
            )

        task_id = data["task_id"]
        print(f"[MiniMax] I2V task: {task_id}")
        return task_id

    def check_status(self, task_id: str) -> dict:
        """Check video generation task status.

        Returns:
            dict with keys: status, file_id (if Success), video_width, video_height
        """
        resp = self.session.get(
            f"{self.BASE_URL}/query/video_generation",
            params={"task_id": task_id},
            timeout=30,
        )
        return resp.json()

    def get_download_url(self, file_id: str) -> str:
        """Get download URL for completed video.

        Returns:
            Direct download URL (expires in ~1 hour).
        """
        resp = self.session.get(
            f"{self.BASE_URL}/files/retrieve",
            params={"file_id": file_id},
            timeout=30,
        )
        data = resp.json()
        return data["file"]["download_url"]

    def wait_and_download(
        self,
        task_id: str,
        output_path: Path,
        poll_interval: int = 10,
        max_wait: int = 600,
    ) -> Path:
        """Poll task status and download result when ready.

        Args:
            task_id: Task ID from generate_* methods.
            output_path: Where to save the MP4 file.
            poll_interval: Seconds between status checks (default 10).
            max_wait: Maximum wait time in seconds (default 600).

        Returns:
            Path to downloaded video file.
        """
        elapsed = 0
        while elapsed < max_wait:
            result = self.check_status(task_id)
            status = result.get("status", "Unknown")

            if status == "Success":
                file_id = result["file_id"]
                print(f"[MiniMax] Done! Downloading file_id={file_id}...")
                url = self.get_download_url(file_id)
                video_resp = requests.get(url, timeout=120)
                output_path.write_bytes(video_resp.content)
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"[MiniMax] Saved: {output_path} ({size_mb:.1f} MB)")
                return output_path

            elif status == "Fail":
                error_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
                raise RuntimeError(f"MiniMax generation failed: {error_msg}")

            else:
                print(f"[MiniMax] Status: {status} ({elapsed}s elapsed)")
                time.sleep(poll_interval)
                elapsed += poll_interval

        raise TimeoutError(f"MiniMax task {task_id} timed out after {max_wait}s")


def generate_avatar_segments(
    client: MiniMaxClient,
    prompts: list[str],
    reference_image_url: str,
    output_dir: Path,
) -> list[Path]:
    """Generate multiple 6-second avatar video segments.

    Args:
        client: MiniMaxClient instance.
        prompts: List of scene descriptions for each segment.
        reference_image_url: Public URL to face reference photo.
        output_dir: Directory to save segments.

    Returns:
        List of paths to downloaded MP4 segments.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = []

    # Submit all tasks
    tasks = []
    for i, prompt in enumerate(prompts):
        try:
            task_id = client.generate_avatar_video(prompt, reference_image_url)
            tasks.append((i, task_id))
            time.sleep(2)  # Brief pause between submissions
        except Exception as e:
            print(f"[MiniMax] Segment {i} submission failed: {e}")

    # Wait and download all
    for i, task_id in tasks:
        try:
            output_path = output_dir / f"avatar_{i:02d}.mp4"
            client.wait_and_download(task_id, output_path)
            segments.append(output_path)
        except Exception as e:
            print(f"[MiniMax] Segment {i} download failed: {e}")

    return segments


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(Path("/opt/ai-video/config/.env"))

    client = MiniMaxClient()

    # Quick test: generate one S2V-01 video
    if len(sys.argv) > 1:
        image_url = sys.argv[1]
    else:
        print("Usage: python3 minimax_api.py <reference_image_url> [prompt]")
        sys.exit(1)

    prompt = sys.argv[2] if len(sys.argv) > 2 else (
        "A man in a beige shirt sits in a modern office, "
        "looks at the camera, and speaks confidently about business strategy. "
        "[Static shot]"
    )

    task_id = client.generate_avatar_video(prompt, image_url)
    output = client.wait_and_download(task_id, Path("/opt/ai-video/output/test_minimax.mp4"))
    print(f"Test video: {output}")
