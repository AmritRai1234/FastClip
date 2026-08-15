# FastClip

AI-powered YouTube Shorts clipping engine — fast, quality-first, built for scale.

## Vision

FastClip turns long-form YouTube content into viral Shorts automatically. Give it a video URL, get back ready-to-post vertical clips with curated captions and highlights. The engine optimizes for **speed** (parallel pipeline, CPU-efficient) and **quality** (smart moment selection, polished output), not just "good enough."

This is the production-grade successor to ClipME — same core idea, rebuilt from the ground up with proper architecture, testing, and a clear path to monetization.

## What Makes It Different

| ClipME (v0) | FastClip (v1) |
|---|---|
| Monolithic script | Modular pipeline architecture |
| Best-effort output | Quality gates at every stage |
| Hardcoded config | CLI + config file + env |
| No tests | Full test coverage |
| Personal tool | Product (free tier → paid) |
| ~1K LOC, single file | Proper package, typed, documented |

## Pipeline

```
YouTube URL → Download (yt-dlp) → Transcribe (Whisper) → Score Moments (LLM)
→ Select Best Clips → Extract + Resize (MoviePy/OpenCV) → Caption Overlay → Output (MP4)
```

Each stage is a pluggable module — swap Whisper for a faster model, swap the LLM scorer, add new output formats without touching the rest.

## Tech Stack

- **Python 3.11+** — typed, async where it matters
- **yt-dlp** — video download + metadata
- **Whisper** (faster-whisper) — transcription, CPU-optimized
- **DeepSeek / OpenRouter** — moment scoring, caption generation
- **MoviePy + OpenCV** — frame extraction, resize, compositing
- **Click / Rich** — CLI and terminal UI

## Project Structure

```
fast-clip/
├── src/fast_clip/      # Package source
│   ├── cli.py          # Entry point
│   ├── pipeline/       # Pipeline stages (download, transcribe, score, render)
│   ├── models/         # Data models (Clip, Video, Config)
│   └── utils/          # Shared utilities
├── tests/              # pytest
├── docs/               # Architecture, API docs
├── pyproject.toml      # Build config
└── README.md           # This file
```

## Monetization Path

1. **Free tier**: 5 clips/day, watermark, standard quality
2. **Pro tier** ($): unlimited clips, no watermark, HD output, priority queue
3. **API tier** ($$): REST API access for integrations, batch processing
4. **Enterprise** ($$$): custom branding, dedicated infra, SLAs

License key validation via local activation + periodic phone-home. Simple, not draconian.

## Roadmap

- [ ] v0.1 — Core pipeline scaffolding, download + transcribe stage
- [ ] v0.2 — LLM moment scoring, clip selection
- [ ] v0.3 — Render pipeline, vertical crop, caption overlay
- [ ] v0.4 — CLI with progress bars, config file
- [ ] v0.5 — Free/pro tier system, watermarking
- [ ] v1.0 — Public launch, docs site, distribution

## Start

```bash
git clone <repo>
cd fast-clip
pip install -e ".[dev]"
fast-clip --help
```