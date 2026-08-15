"""CLI entry point for FastClip."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from fast_clip.pipeline.downloader import Downloader
from fast_clip.pipeline.orchestrator import run_pipeline
from fast_clip.pipeline.scorer import Scorer
from fast_clip.pipeline.transcriber import MODEL_SIZES, MODEL_SIZE_MB, Transcriber
from fast_clip.utils.llm import DEFAULT_MODEL

console = Console()

OUTPUT_FORMATS = ("txt", "srt", "json")

# Default output directory (project-local; gitignored).
DEFAULT_OUTPUT_DIR = "output"


def _is_url(value: str) -> bool:
    """Heuristic: treat http(s)/www-prefixed input as a URL, else a file path."""
    return value.startswith(("http://", "https://", "www."))


def _download_with_progress(
    url: str, output_dir: Path, max_height: int | None
) -> Path:
    """Download a video with a Rich progress bar; return the local filepath."""
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    dl = Downloader(output_dir=output_dir, max_height=max_height, quiet=True)
    task_id: list[int] = []

    def hook(d: dict) -> None:
        status = d.get("status", "")
        if status == "downloading":
            if not task_id:
                task_id.append(
                    progress.add_task(
                        "Downloading",
                        total=d.get("total_bytes") or d.get("total_bytes_estimate") or None,
                    )
                )
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            progress.update(task_id[0], total=total, completed=d.get("downloaded_bytes", 0))
        elif status == "finished" and task_id:
            progress.update(task_id[0], completed=progress.tasks[task_id[0]].total)

    dl.progress_callback = hook
    with progress:
        meta = dl.download(url)
    return meta.filepath


def _display_clips(
    transcript_lang: str,
    transcript_prob: float,
    result,  # ScoreResult
    saved_path: Path | None = None,
) -> None:
    """Render clip recommendations as a Rich table and auto-save."""
    clips = result.clips
    tokens = result.tokens_used

    console.print(
        f"\n[bold]AI Clip Analysis[/bold] — language [green]{transcript_lang}[/green] "
        f"({transcript_prob:.0%})"
    )
    if tokens:
        console.print(
            f"[dim]LLM tokens: {tokens['input']:,} in / {tokens['output']:,} out "
            f"({tokens['total']:,} total)[/dim]"
        )
    if saved_path:
        console.print(f"[dim]Saved: {saved_path}[/dim]")

    table = Table(title="Recommended Clips")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Start", style="cyan", justify="right")
    table.add_column("End", style="cyan", justify="right")
    table.add_column("Dur", style="dim", justify="right")
    table.add_column("Title", style="bold")
    table.add_column("Hook")

    for i, c in enumerate(clips, 1):
        dur = c.end - c.start
        table.add_row(
            str(i),
            f"{c.start:.1f}s",
            f"{c.end:.1f}s",
            f"{dur:.0f}s",
            c.title,
            c.hook,
        )

    console.print(table)

    # Show reasoning for each clip.
    console.print("\n[bold]Why these clips?[/bold]")
    for i, c in enumerate(clips, 1):
        console.print(f"  [cyan]#{i}[/cyan] {c.reasoning}")


def _resolve_to_transcript(
    path: Path,
    *,
    model: str,
    language: str | None,
    vad: bool,
    transcribe_dir: Path,
) -> tuple:
    """Given a file path, return a (Transcript, media_path) tuple.

    If the file is a JSON transcript, load it. If it's a video/audio file,
    transcribe it first. If it's a txt/srt, parse back to segments.
    """
    path = Path(path)
    if not path.exists():
        raise click.BadParameter(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        # Try loading as a saved transcript JSON.
        import json
        from fast_clip.pipeline.transcriber import Segment, Transcript

        data = json.loads(path.read_text())
        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"])
            for s in data.get("segments", [])
        ]
        transcript = Transcript(
            language=data.get("language", "unknown"),
            language_probability=data.get("language_probability", 1.0),
            duration=data.get("duration", 0.0),
            segments=segments,
        )
        return transcript, path

    if suffix in (".txt", ".srt"):
        raise click.BadParameter(
            "Scoring from .txt/.srt is not yet supported — "
            "use the JSON transcript (transcribe with -f json) for best results."
        )

    # Treat as media file: transcribe it.
    console.print(
        f"[dim]Whisper model: [bold]{model}[/bold] (~{MODEL_SIZE_MB[model]} MB)[/dim]"
    )
    tr = Transcriber(model_size=model, vad_filter=vad)
    with console.status(f"[bold green]Transcribing {path.name}...[/bold green]"):
        transcript = tr.transcribe(path, language=language)
    console.print(
        f"[dim]Transcribed: {len(transcript.segments)} segments "
        f"({transcript.language}, {transcript.language_probability:.0%})[/dim]"
    )
    # Save JSON so the user has it for future use.
    out_path = transcript.save(transcribe_dir / f"{path.stem}.json", "json")
    console.print(f"[dim]Saved transcript: {out_path}[/dim]")
    return transcript, path


@click.group()
@click.version_option()
def main() -> None:
    """FastClip — AI-powered YouTube Shorts clipping engine."""
    pass


@main.command()
@click.argument("media", metavar="FILE_OR_URL")
@click.option(
    "--model", "-m",
    default="base",
    type=click.Choice(MODEL_SIZES),
    show_default=True,
    help="Whisper model size (speed vs accuracy).",
)
@click.option(
    "--language", "-l",
    default=None,
    help="Force a language (ISO code, e.g. 'en'). Auto-detect if omitted.",
)
@click.option(
    "--vad/--no-vad",
    default=True,
    show_default=True,
    help="Skip silence via voice-activity detection (disable for music-heavy video).",
)
@click.option(
    "--format", "-f", "fmt",
    default="txt",
    type=click.Choice(("txt", "srt", "json", "all")),
    show_default=True,
    help="Output format(s) to write.",
)
@click.option(
    "--output", "-o",
    default=DEFAULT_OUTPUT_DIR,
    type=click.Path(file_okay=False, path_type=Path),
    show_default=True,
    help="Output directory for transcript files.",
)
@click.option(
    "--max-height",
    default=None,
    type=int,
    help="Resolution cap when input is a URL (e.g. 720). Default: highest.",
)
@click.option(
    "--score/--no-score",
    default=False,
    show_default=True,
    help="After transcribing, run the LLM moment scorer to find the best clips.",
)
@click.option(
    "--llm-model",
    default="deepseek/deepseek-v4-flash",
    show_default=True,
    help="LLM model to use for scoring (via OpenRouter).",
)
@click.option(
    "--llm-temperature",
    default=0.7,
    show_default=True,
    type=float,
    help="LLM temperature for scoring (higher = more creative picks).",
)
def transcribe(
    media: str,
    model: str,
    language: str | None,
    vad: bool,
    fmt: str,
    output: Path,
    max_height: int | None,
    score: bool,
    llm_model: str,
    llm_temperature: float,
) -> None:
    """Transcribe a video/audio file (or YouTube URL) to timestamped text.

    Outputs .txt (readable), .srt (subtitles), and/or .json (structured) —
    the same transcript the moment-scorer stage consumes downstream.

    Pass --score to automatically run the AI moment scorer after transcribing.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # ── Resolve input to a local media file ──
    if _is_url(media):
        console.print(f"[bold]FastClip[/bold] — downloading [cyan]{media}[/cyan]")
        media_path = _download_with_progress(media, output, max_height)
        console.print(f"[dim]Saved: {media_path}[/dim]\n")
    else:
        media_path = Path(media)
        if not media_path.exists():
            raise click.BadParameter(f"Media file not found: {media_path}")

    # ── Transcribe ──
    console.print(
        f"[dim]Model: [bold]{model}[/bold] (~{MODEL_SIZE_MB[model]} MB, "
        f"downloaded automatically on first use)[/dim]"
    )
    tr = Transcriber(model_size=model, vad_filter=vad)
    with console.status(f"[bold green]Transcribing {media_path.name}...[/bold green]"):
        transcript = tr.transcribe(media_path, language=language)

    if not transcript.segments:
        console.print(
            "[yellow]No speech segments found.[/yellow] "
            "If this is music-heavy, retry with --no-vad."
        )
        return

    # ── Save requested format(s) ──
    formats = list(OUTPUT_FORMATS) if fmt == "all" else [fmt]
    if score and "json" not in formats:
        formats.append("json")  # scorer needs the JSON transcript
    stem = media_path.stem
    saved = [transcript.save(output / f"{stem}.{f}", f) for f in formats]

    console.print(
        f"\n[bold]Transcript[/bold] — language [green]{transcript.language}[/green] "
        f"({transcript.language_probability:.0%} confidence), "
        f"{transcript.duration:.0f}s, [bold]{len(transcript.segments)}[/bold] segments"
    )
    for p in saved:
        console.print(f"[dim]Wrote: {p}[/dim]")

    # ── Preview ──
    if not score:
        table = Table(title=f"{stem} — first 20 segments")
        table.add_column("Start", style="cyan", justify="right")
        table.add_column("End", style="cyan", justify="right")
        table.add_column("Text")
        for seg in transcript.segments[:20]:
            table.add_row(f"{seg.start:6.2f}s", f"{seg.end:6.2f}s", seg.text)
        if len(transcript.segments) > 20:
            table.add_row("", "", f"… and {len(transcript.segments) - 20} more")
        console.print(table)

    # ── Score (LLM moment analysis) ──
    if score:
        console.print("\n[bold cyan]Scoring moments with AI...[/bold cyan]")
        scorer = Scorer(model=llm_model, temperature=llm_temperature)
        with console.status(f"[bold green]Analyzing {len(transcript.segments)} segments...[/bold green]"):
            result = scorer.score(transcript)

        if not result.clips:
            console.print("[yellow]No clips identified — try a different video.[/yellow]")
            return

        score_path = output / f"{stem}_clips.json"
        result.save(score_path)
        _display_clips(
            transcript.language, transcript.language_probability,
            result, saved_path=score_path,
        )


@main.command()
@click.argument("input", metavar="TRANSCRIPT_OR_VIDEO")
@click.option(
    "--model", "-m",
    default="base",
    type=click.Choice(MODEL_SIZES),
    show_default=True,
    help="Whisper model (only if input is a video file).",
)
@click.option(
    "--llm-model",
    default="deepseek/deepseek-v4-flash",
    show_default=True,
    help="LLM model for scoring (via OpenRouter).",
)
@click.option(
    "--llm-temperature",
    default=0.7,
    show_default=True,
    type=float,
    help="LLM temperature for scoring (higher = more creative).",
)
@click.option(
    "--vad/--no-vad",
    default=True,
    show_default=True,
    help="Skip silence via VAD (only if input is a video file).",
)
@click.option(
    "--language", "-l",
    default=None,
    help="Force language (only if input is a video file).",
)
@click.option(
    "--output", "-o",
    default=DEFAULT_OUTPUT_DIR,
    type=click.Path(file_okay=False, path_type=Path),
    show_default=True,
    help="Output directory (for intermediates if transcribing).",
)
def score_cmd(
    input: str,
    model: str,
    llm_model: str,
    llm_temperature: float,
    vad: bool,
    language: str | None,
    output: Path,
) -> None:
    """Score a transcript (or video file) and get AI clip recommendations.

    Accepts a transcript JSON file (fastest — transcribe with -f json first)
    or a video/audio file (auto-transcribes, then scores).

    The AI reads the full transcript and returns 3-5 clip ideas with
    suggested titles, hooks, captions, and reasoning.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    transcript, _media_path = _resolve_to_transcript(
        Path(input),
        model=model,
        language=language,
        vad=vad,
        transcribe_dir=output,
    )

    if not transcript.segments:
        console.print("[yellow]No speech segments to score.[/yellow]")
        return

    console.print(f"\n[bold cyan]Scoring with {llm_model}...[/bold cyan]")
    scorer = Scorer(model=llm_model, temperature=llm_temperature)
    with console.status(f"[bold green]Analyzing {len(transcript.segments)} segments...[/bold green]"):
        result = scorer.score(transcript)

    if not result.clips:
        console.print("[yellow]No clips identified — try a different video or lower temperature.[/yellow]")
        return

    score_path = output / f"{Path(input).stem}_clips.json"
    result.save(score_path)
    _display_clips(
        transcript.language, transcript.language_probability,
        result, saved_path=score_path,
    )


@main.command()
@click.argument("url", metavar="URL_OR_FILE")
@click.option("--output", "-o", default=DEFAULT_OUTPUT_DIR, type=click.Path(file_okay=False, path_type=Path), help="Output directory.")
@click.option("--whisper-model", default="base", type=click.Choice(MODEL_SIZES), show_default=True, help="Whisper model size.")
@click.option("--llm-model", default=DEFAULT_MODEL, show_default=True, help="LLM model for segmentation/clip planning (OpenRouter).")
@click.option("--max-height", default=None, type=int, help="Resolution cap when input is a URL (e.g. 720).")
@click.option("--cookies", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Netscape cookies.txt from a logged-in browser (unlocks 1080p+).")
@click.option("--cookies-from-browser", default=None, help="Read YouTube login from a browser (e.g. chrome, firefox) instead of a cookies file.")
@click.option("--no-pan", is_flag=True, help="Disable face-following pan (use static center crop).")
@click.option("--language", "-l", default=None, help="Force source language (ISO code).")
def clip(
    url: str,
    output: Path,
    whisper_model: str,
    llm_model: str,
    max_height: int | None,
    cookies: Path | None,
    cookies_from_browser: str | None,
    no_pan: bool,
    language: str | None,
) -> None:
    """Turn a YouTube video (or local file) into captioned 9:16 Shorts.

    Runs the full pipeline: download -> transcribe -> segment naturally -> plan
    Shorts -> render vertical with face-following pan and word-by-word captions.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def _progress(d: dict) -> None:
        stage, msg = d["stage"], d["message"]
        if stage == "render":
            console.print(f"  [{d['current']}/{d['total']}] {msg}")
        elif stage == "plan":
            console.print(f"  [plan] {msg}")
        elif stage == "done":
            pass
        else:
            console.print(f"[dim]{stage}: {msg}[/dim]")

    console.print(f"[bold]FastClip[/bold] — processing [cyan]{url}[/cyan]")
    results = run_pipeline(
        url,
        output_dir=output,
        whisper_model=whisper_model,
        llm_model=llm_model,
        max_height=max_height,
        cookies_file=cookies,
        cookies_from_browser=cookies_from_browser,
        pan=not no_pan,
        language=language,
        progress=_progress,
    )
    console.print(f"\n[bold green]Done — {len(results)} Shorts in {output / 'shorts'}/[/bold green]")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option("--port", default=8900, type=int, show_default=True, help="Port.")
@click.option("--jobs-dir", default="output/jobs", type=click.Path(file_okay=False, path_type=Path), show_default=True, help="Where job outputs are written.")
def serve(host: str, port: int, jobs_dir: Path) -> None:
    """Run the FastClip web API (backed by the full clip pipeline)."""
    import uvicorn

    from fast_clip.api.routes import create_app
    from fast_clip.pipeline.bgutil import ensure_bgutil_server

    if ensure_bgutil_server():
        console.print("[dim]bgutil PO-token server: running (1080p YouTube downloads enabled)[/dim]")
    else:
        console.print("[yellow]bgutil server not available — YouTube downloads fall back to 360p.[/yellow]")

    app = create_app(jobs_dir)
    console.print(f"[bold]FastClip API[/bold] — http://{host}:{port}  (jobs in {jobs_dir})")
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command()
def config() -> None:
    """Show or set configuration."""
    console.print("[yellow]Config management not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()