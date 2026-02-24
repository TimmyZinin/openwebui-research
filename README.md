# AI Video Factory v2 — OpenClaw + HeyGen

Content factory on a regular VPS: OpenClaw (221K stars, autonomous AI agent) automates HeyGen via browser, edits videos with b-roll, publishes to 14 platforms. Uses cheap Chinese LLMs (DeepSeek $0.14/1M or GLM-4.7 free).

## Live Presentation

[timmyzinin.github.io/openwebui-research](https://timmyzinin.github.io/openwebui-research/)

## Key Findings

- **OpenClaw**: Standalone AI agent (NOT a Claude wrapper). 14+ LLM providers, Telegram-first, browser tools, persistent memory
- **LLM**: DeepSeek V3 ($0.14/1M input) or GLM-4.7-Flash (FREE) — work as drop-in providers
- **HeyGen Creator $29/mo**: "Unlimited" Avatar III videos. Photo Avatar: no ID verification for AI characters
- **Video editing**: MoviePy v2 + FFmpeg on CPU. 60s video in 3-8 min
- **B-roll**: Pexels API (free, 150K+ videos) + Gemini Flash (500 images/day free)
- **Total budget**: $34-42/mo (was $344-564/mo in v1). Savings: 8-16x

## Architecture

```
Tim (Telegram) → OpenClaw (VPS, $5/mo)
                     ├── LLM: DeepSeek V3 / GLM-4.7 / Qwen3
                     ├── Browser Tool → HeyGen Web (Avatar III, $29/mo)
                     ├── Shell Tool: MoviePy + FFmpeg (no GPU)
                     ├── Pexels API + Gemini (b-roll, free)
                     └── rclone SFTP → RUVDS (Yuki Agent) → 14 social networks
```

## LLM Comparison

| Model | Input $/1M | Tool Calling | Russian | 500 videos/mo |
|-------|-----------|-------------|---------|---------------|
| GLM-4.7-Flash | $0 | 87.4 SOTA | OK | $0/mo |
| DeepSeek V3 | $0.14 | 81.5% | Good | ~$2/mo |
| Qwen3-Max | $1.20 | 96.5% | Best | ~$7/mo |

## VPS

Contabo Cloud VPS 10: 4 vCPU / 8 GB RAM / 75 GB NVMe / Ubuntu 24.04 / EU

## Files

- `setup_vps.sh` — VPS bootstrap script (Node.js 20, OpenClaw, Python, FFmpeg)
- `openclaw_config.yaml` — OpenClaw agent configuration
- `scripts/video_pipeline.py` — main video production pipeline
- `scripts/heygen_api.py` — HeyGen API client (Photo Avatar + video generation)
- `scripts/upload_to_ruvds.sh` — upload finished videos to RUVDS for publishing
- `lisa_photos/` — AI-generated Lisa Solovyova photos (Nano Banana)
- `index.html` — interactive presentation (warm theme)
