# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered podcast editing platform. Users upload audio, which gets transcribed via AssemblyAI, analyzed by Claude for edit decisions (filler words, long pauses, etc.), then edited with pydub. Audio enhancement is done externally (e.g. Adobe Enhance Speech) between the cuts and final export steps.

## Architecture

**Frontend:** Single-file React SPA (`index.html`) using CDN-loaded React 18, Tailwind CSS, and Babel Standalone for JSX. No build step — served as static HTML on GitHub Pages.

**Backend:** Flask REST API (`backend/app.py`) deployed on Railway. Orchestrates AssemblyAI transcription, Claude analysis, and pydub audio editing.

**Flow (auto-enhance):** Frontend uploads audio → Backend sends to AssemblyAI → Backend polls for transcript → Backend sends compact prompt to Claude → Returns edit decisions → Backend applies cuts with pydub → Backend sends to Adobe Enhance Speech API → Backend finalizes (stereo + peak-normalize) → Returns edited MP3.

**Flow (manual fallback):** Same as above, but if `ADOBE_ENHANCE_TOKEN` is not set or Adobe API fails, the pipeline stops after cuts. User downloads intermediate WAV → Enhances externally (Adobe Enhance Speech) → Re-uploads enhanced file → Backend finalizes → Returns edited MP3.

The backend API URL is hardcoded in the frontend: `https://ai-podcast-editor-production.up.railway.app`

## Key Files

- `index.html` — Entire frontend (React components, state management, UI)
- `backend/app.py` — Entire backend (Flask routes, AI integration, audio processing)
- `backend/requirements.txt` — Python dependencies
- `backend/railway.json` — Railway deployment config
- `backend/Procfile` — Gunicorn process definition

## Development Commands

### Backend (local)
```bash
cd backend
pip install -r requirements.txt
export ASSEMBLYAI_API_KEY="your_key"
export CLAUDE_API_KEY="your_key"
python app.py  # http://localhost:5000
```

### Frontend
No build — open `index.html` in browser or serve with any static file server.

## Backend API Routes

- `GET /` — Health check
- `GET /api/status` — Config validation
- `POST /api/upload` — Upload audio, starts AssemblyAI transcription, returns `transcript_id`
- `GET /api/transcription-status/<transcript_id>` — Poll transcription progress
- `POST /api/process` — Start Claude analysis (async), returns `job_id`
- `GET /api/process-status/<job_id>` — Poll analysis results
- `POST /api/edit-audio` — Apply cuts to audio (phase 1), returns `job_id`
- `GET /api/edit-audio-status/<job_id>` — Poll edit status (`cutting` → `enhancing` → `finalizing` → `completed`, or `cutting` → `cuts_completed` if manual)
- `GET /api/edit-audio-download/<job_id>` — Download intermediate WAV (when `cuts_completed`) or final MP3 (when `completed`)
- `POST /api/upload-enhanced` — Upload enhanced audio file with `job_id`, starts finalization (phase 2)

## Key Technical Details

- **Two-phase editing:** Phase 1 applies cuts and exports WAV. If `ADOBE_ENHANCE_TOKEN` is set, enhancement and finalization happen automatically. Otherwise, user enhances externally and re-uploads. Phase 2 finalizes (stereo + normalize + MP3).
- **Async processing:** Claude analysis runs in a background thread; frontend polls `/api/process-status/<job_id>` every 3 seconds
- **Compact prompts:** Filler words and pauses are pre-detected in Python before sending to Claude, keeping the prompt small to avoid Railway timeouts
- **Audio storage:** Uploaded files saved in `/tmp` with `transcript_id` as filename
- **Thread safety:** Uses `threading.Lock()` for the jobs dictionary — requires threaded gunicorn workers

## Environment Variables (Railway)

- `ASSEMBLYAI_API_KEY` — AssemblyAI transcription API key
- `CLAUDE_API_KEY` — Anthropic Claude API key
- `ADOBE_ENHANCE_TOKEN` — (optional) Bearer token from Adobe session for auto-enhance. When set, the pipeline runs end-to-end. When not set, stops at cuts for manual enhancement.
