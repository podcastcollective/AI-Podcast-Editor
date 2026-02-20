# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered podcast editing platform. Users upload audio, which gets transcribed via AssemblyAI, analyzed by Claude for edit decisions (filler words, long pauses, etc.), then edited with pydub.

## Architecture

**Frontend:** Single-file React SPA (`index.html`) using CDN-loaded React 18, Tailwind CSS, and Babel Standalone for JSX. No build step — served as static HTML on GitHub Pages.

**Backend:** Flask REST API (`backend/app.py`) deployed on Railway. Orchestrates AssemblyAI transcription, Claude analysis, and pydub audio editing.

**Flow:** Frontend uploads audio → Backend sends to AssemblyAI → Backend polls for transcript → Backend sends compact prompt to Claude (with JSON prefill trick) → Returns edit decisions → Frontend displays for human review → Backend applies cuts with pydub → Returns edited MP3.

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
- `POST /api/edit-audio` — Apply edit decisions to audio, returns edited MP3

## Key Technical Details

- **Claude JSON prefill:** Backend prefills assistant response with `"["` to force JSON array output, avoiding markdown/text wrapper issues
- **Async processing:** Claude analysis runs in a background thread; frontend polls `/api/process-status/<job_id>` every 3 seconds
- **Compact prompts:** Filler words and pauses are pre-detected in Python before sending to Claude, keeping the prompt small to avoid Railway timeouts
- **Audio storage:** Uploaded files saved in `/tmp` with `transcript_id` as filename
- **Thread safety:** Uses `threading.Lock()` for the jobs dictionary — requires threaded gunicorn workers

## Environment Variables (Railway)

- `ASSEMBLYAI_API_KEY` — AssemblyAI transcription API key
- `CLAUDE_API_KEY` — Anthropic Claude API key
