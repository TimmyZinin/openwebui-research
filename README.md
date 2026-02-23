# AI Video Factory v2 — Claude Code + HeyGen

Content factory on a regular VPS: Claude Code automates HeyGen via browser, edits videos with b-roll, publishes to 14 platforms.

## Live Presentation

[timmyzinin.github.io/openwebui-research](https://timmyzinin.github.io/openwebui-research/)

## Key Findings

- **HeyGen Creator $29/mo**: "Unlimited" Avatar III videos via web interface. No GPU needed
- **Photo Avatar**: No identity verification for AI-generated characters (Lisa)
- **Claude Code on VPS**: Headless via `-p` flag + Playwright MCP for browser automation
- **OpenClaw**: Ready-made Telegram → Claude Code wrapper for 24/7 operation
- **Video editing**: MoviePy v2 + FFmpeg on CPU. 60s video renders in 3-8 min
- **B-roll**: Pexels API (free, 150K+ videos) + Gemini Flash (500 images/day free)
- **Total budget**: $50-80/mo (was $344-564/mo in v1)

## Architecture

```
Tim (Telegram) → OpenClaw / Claude Code (VPS)
                     ├── Playwright → HeyGen Web (Avatar III, $29/mo)
                     ├── MoviePy + FFmpeg (video editing, no GPU)
                     ├── Pexels API + Gemini (b-roll, free)
                     └── rclone SFTP → RUVDS (Yuki Agent) → 14 social networks
```

## Pipeline (per video)

1. Tim sends topic via Telegram
2. Claude generates script (Haiku 4.5)
3. Playwright automates HeyGen: avatar + script → MP4
4. B-roll: Pexels stock + Gemini AI images
5. MoviePy: 2s talking head → 2s b-roll → transitions → music
6. rclone → RUVDS → Yuki Agent → 14 platforms

## Files

- `index.html` — interactive presentation (warm theme)
- Full research: `memory/openwebui_ovh_research.md`
