# Open WebUI + OVH Cloud: AI Video Factory

Deep research report on integrating Open WebUI with GPU cloud for automated video shorts generation.

## Live Presentation

[timmyzinin.github.io/openwebui-research](https://timmyzinin.github.io/openwebui-research/)

## Key Findings

- **Open WebUI** (125K+ stars): ideal frontend with Ollama, LiteLLM, Tools/Pipelines, RAG
- **OVH Cloud**: too expensive ($730+/mo). **Vast.ai RTX 4090 ($234/mo)** recommended
- **LLM**: Qwen3 14B — best for Russian + tool calling on 24GB VRAM
- **Video**: HeyGen API ($99/mo) for premium + EchoMimicV3 (self-hosted) for bulk
- **Transfer**: rclone (SFTP) + webhook — free, reliable, simple
- **Total budget**: $344/mo for full stack

## Architecture

```
GPU VPS (Vast.ai) → Open WebUI + Ollama + EchoMimicV3 + ffmpeg
                  → rclone SFTP → RUVDS (Yuki Agent) → 14 social networks
Cloud APIs        → HeyGen, Groq, DeepSeek, Gemini (b-roll)
```

## Files

- `index.html` — interactive presentation (dark/neon theme)
- Full research report: `memory/openwebui_ovh_research.md`
