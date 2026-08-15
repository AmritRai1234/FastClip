"use client";

import { useCallback, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8900";

interface Short {
  title: string;
  hook: string;
  duration: number;
  section: number;
  section_title: string;
  filename: string;
  url: string;
}

interface Job {
  id: string;
  url: string;
  status: string;
  stage: string;
  current: number;
  total: number;
  message: string;
  shorts: Short[];
  error: string | null;
}

const STAGES: { key: string; label: string }[] = [
  { key: "download", label: "Download" },
  { key: "transcribe", label: "Transcribe" },
  { key: "segment", label: "Find sections" },
  { key: "plan", label: "Plan Shorts" },
  { key: "render", label: "Render" },
];

const BROWSERS = ["chrome", "firefox", "chromium", "edge", "brave", "opera", "vivaldi"];

const NOISE = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`;

function fmtDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function viralityScore(short: Short): number {
  let s = 58;
  const d = short.duration;
  if (d >= 25 && d <= 45) s += 22;
  else if (d >= 15 && d <= 60) s += 10;
  else if (d < 8 || d > 90) s -= 22;
  if (short.hook && short.hook.trim().length > 0) s += 12;
  return Math.max(0, Math.min(100, s));
}

function scoreClass(score: number): string {
  if (score >= 80) return "bg-[#ffcb5c] text-black";
  if (score >= 50) return "bg-[#ffcb5c] text-[#1a1200]";
  return "bg-white/10 text-zinc-400";
}

export default function Home() {
  const [url, setUrl] = useState("");
  const [whisperModel, setWhisperModel] = useState("base");
  const [pan, setPan] = useState(true);
  const [browser, setBrowser] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [mode, setMode] = useState<"url" | "upload">("url");
  const fileInput = useRef<HTMLInputElement>(null);

  const poll = useCallback(async (id: string) => {
    try {
      const r = await fetch(`${API}/api/jobs/${id}`);
      const j: Job = await r.json();
      setJob(j);
      if (j.status === "running" || j.status === "queued") {
        setTimeout(() => poll(id), 2000);
      }
    } catch {
      setTimeout(() => poll(id), 3000);
    }
  }, []);

  const startFromUrl = async () => {
    if (!url.trim()) return;
    setError("");
    setSubmitting(true);
    setJob(null);
    try {
      const r = await fetch(`${API}/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim(), whisper_model: whisperModel, pan, cookies_from_browser: browser || null }),
      });
      if (!r.ok) throw new Error(`Server responded ${r.status}`);
      const j: Job = await r.json();
      setJob(j);
      poll(j.id);
    } catch (e) {
      setError(`Failed to start: ${e instanceof Error ? e.message : e}`);
    } finally {
      setSubmitting(false);
    }
  };

  const uploadFile = async (file: File) => {
    setError("");
    setSubmitting(true);
    setJob(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("whisper_model", whisperModel);
      fd.append("pan", String(pan));
      const r = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`Server responded ${r.status}`);
      const j: Job = await r.json();
      setJob(j);
      poll(j.id);
    } catch (e) {
      setError(`Upload failed: ${e instanceof Error ? e.message : e}`);
    } finally {
      setSubmitting(false);
    }
  };

  const running = job !== null && (job.status === "running" || job.status === "queued");
  const progress = job && job.total > 0 ? Math.round((job.current / job.total) * 100) : 0;
  const hasShorts = job !== null && job.shorts.length > 0;
  const stageIndex = job ? STAGES.findIndex((s) => s.key === job.stage) : -1;

  return (
    <div className="relative min-h-full bg-black text-white">
      {/* Film grain overlay */}
      <div className="pointer-events-none fixed inset-0 opacity-[0.035]" style={{ backgroundImage: NOISE }} />

      {/* Header */}
      <header className="relative mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#ffcb5c] text-[#1a1200] text-sm">⚡</span>
          <span className="text-[15px] font-semibold tracking-tight">FastClip</span>
        </div>
        <span className="text-xs text-zinc-500">AI Shorts, from any video</span>
      </header>

      <main className="relative mx-auto max-w-6xl px-6 pb-24">
        {/* Hero */}
        <section className="mt-12 text-center sm:mt-16">
          <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3.5 py-1.5 text-[11px] uppercase tracking-[0.18em] text-zinc-300">
            <span className="h-1.5 w-1.5 rounded-full bg-[#ffcb5c]" />
            AI video clipping
          </span>
          <h1 className="mx-auto mt-6 max-w-3xl text-5xl font-semibold leading-[1.04] tracking-tighter sm:text-6xl">
            Turn long videos into viral Shorts.
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-base text-zinc-400">
            Paste a link or drop a file. AI finds the best moments and cuts them into captioned 9:16 Shorts.
          </p>
        </section>

        {/* Source */}
        <section className="mx-auto mt-12 max-w-2xl">
          <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-5 shadow-2xl shadow-black/50 backdrop-blur">
            {/* Mode tabs */}
            <div className="mb-5 flex gap-1 rounded-full bg-white/[0.04] p-1">
              <button
                onClick={() => setMode("url")}
                className={`flex-1 rounded-full px-4 py-2 text-sm font-medium transition ${
                  mode === "url" ? "bg-white/10 text-white" : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                YouTube URL
              </button>
              <button
                onClick={() => setMode("upload")}
                className={`flex-1 rounded-full px-4 py-2 text-sm font-medium transition ${
                  mode === "upload" ? "bg-white/10 text-white" : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                Upload video
              </button>
            </div>

            {mode === "url" ? (
              <div className="flex flex-col gap-3 sm:flex-row">
                <input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && startFromUrl()}
                  placeholder="Paste a YouTube link…"
                  className="flex-1 rounded-full border border-white/10 bg-black/40 px-5 py-3.5 text-sm outline-none transition placeholder:text-zinc-600 focus:border-[#ffcb5c]/50"
                />
                <button
                  onClick={startFromUrl}
                  disabled={submitting || running || !url.trim()}
                  className="rounded-full bg-[#ffcb5c] px-7 py-3.5 text-sm font-semibold text-[#1a1200] shadow-[0_0_28px_rgba(255,203,92,0.35)] transition hover:bg-[#ffd88a] disabled:opacity-40 disabled:shadow-none"
                >
                  {submitting ? "Starting…" : "Generate Shorts"}
                </button>
              </div>
            ) : (
              <div
                onClick={() => fileInput.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragActive(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) uploadFile(f);
                }}
                className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition ${
                  dragActive ? "border-[#ffcb5c]/70 bg-[#ffcb5c]/[0.06]" : "border-white/10 bg-black/30 hover:border-white/25"
                }`}
              >
                <span className="text-sm font-medium text-zinc-200">Drop a video here, or click to browse</span>
                <span className="mt-1.5 text-xs text-zinc-500">Full quality — no YouTube, no download, no login</span>
                <input
                  ref={fileInput}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) uploadFile(f);
                    e.target.value = "";
                  }}
                />
              </div>
            )}

            {/* Options */}
            <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-white/5 pt-4 text-xs text-zinc-400">
              <label className="flex items-center gap-2">
                Whisper
                <select
                  value={whisperModel}
                  onChange={(e) => setWhisperModel(e.target.value)}
                  className="rounded-md border border-white/10 bg-black/40 px-2 py-1 outline-none"
                >
                  <option value="tiny">tiny</option>
                  <option value="base">base</option>
                  <option value="small">small</option>
                  <option value="medium">medium</option>
                </select>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={pan} onChange={(e) => setPan(e.target.checked)} className="accent-[#ffcb5c]" />
                Face-following pan
              </label>
              <label className="flex items-center gap-2" title="Log into YouTube in this browser, then pick it here for 1080p">
                YouTube login
                <select
                  value={browser}
                  onChange={(e) => setBrowser(e.target.value)}
                  className="rounded-md border border-white/10 bg-black/40 px-2 py-1 outline-none"
                >
                  <option value="">None (360p)</option>
                  {BROWSERS.map((b) => (
                    <option key={b} value={b}>
                      {b}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          {error && <div className="mt-4 rounded-xl border border-red-900/60 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div>}
        </section>

        {/* Progress */}
        {job && (
          <section className="mx-auto mt-12 max-w-3xl">
            <div className="flex items-center justify-between">
              {STAGES.map((s, i) => {
                const done = job.status === "done" || (stageIndex >= 0 && i < stageIndex) || job.stage === "done";
                const active = stageIndex === i && job.status !== "done";
                return (
                  <div key={s.key} className="flex flex-1 items-center">
                    <div className="flex flex-col items-center gap-1.5">
                      <div
                        className={`flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-semibold transition ${
                          done ? "bg-[#ffcb5c] text-[#1a1200]" : active ? "bg-[#ffcb5c]/30 text-white ring-2 ring-[#ffcb5c]" : "bg-white/10 text-zinc-500"
                        }`}
                      >
                        {done ? "✓" : i + 1}
                      </div>
                      <span className={`text-[10px] ${active ? "text-white" : "text-zinc-500"}`}>{s.label}</span>
                    </div>
                    {i < STAGES.length - 1 && <div className={`mx-1 mb-4 h-px flex-1 ${done ? "bg-[#ffcb5c]" : "bg-white/10"}`} />}
                  </div>
                );
              })}
            </div>

            <div className="mt-6 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-[#ffcb5c] transition-all"
                style={{ width: `${job.stage === "done" ? 100 : progress}%` }}
              />
            </div>
            <p className="mt-3 truncate text-center text-xs text-zinc-500">{job.error ? job.error : job.message}</p>
          </section>
        )}

        {/* Results */}
        {hasShorts && (
          <section className="mt-14">
            <div className="flex items-end justify-between">
              <h2 className="text-2xl font-semibold tracking-tight">{job.shorts.length} Shorts ready</h2>
              <span className="text-xs text-zinc-500">Click to preview · ⬇ to download</span>
            </div>
            <div className="mt-6 grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-4">
              {job.shorts.map((s) => {
                const sc = viralityScore(s);
                return (
                  <div key={s.filename} className="group overflow-hidden rounded-2xl border border-white/10 bg-white/[0.03] transition hover:border-white/25">
                    <div className="relative">
                      <video src={`${API}${s.url}`} controls preload="metadata" className="aspect-[9/16] w-full bg-black object-cover" />
                      <span className="absolute right-2 top-2 rounded-md bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-white">
                        {fmtDuration(s.duration)}
                      </span>
                      <span className={`absolute left-2 top-2 flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold ${scoreClass(sc)}`}>
                        {sc}
                      </span>
                    </div>
                    <div className="p-3">
                      <h3 className="truncate text-sm font-medium text-white" title={s.title}>
                        {s.title}
                      </h3>
                      {s.hook && <p className="mt-1 line-clamp-2 text-xs text-zinc-400">{s.hook}</p>}
                      <div className="mt-2 flex items-center justify-between">
                        <span className="truncate text-[10px] text-zinc-500">{s.section_title || `Section ${s.section}`}</span>
                        <a
                          href={`${API}${s.url}`}
                          download
                          className="shrink-0 rounded-md bg-white/10 px-2 py-1 text-xs text-zinc-200 transition hover:bg-white/20"
                        >
                          ⬇
                        </a>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
