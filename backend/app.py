"""
Flask Backend API for AI Podcast Editor
Handles file uploads, transcription, Claude analysis, and audio editing.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import re
import time
import threading
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import anthropic
import requests

app = Flask(__name__)

# CORS: allow all origins without credentials (required for GitHub Pages -> Railway)
CORS(app)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}

# Get API keys from environment variables
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')

if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
    print("WARNING: API keys not set. Please set ASSEMBLYAI_API_KEY and CLAUDE_API_KEY environment variables")

# In-memory job store for async Claude analysis.
# Gunicorn must use threads (not multiple processes) so this dict is shared.
_jobs: dict = {}
_jobs_lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def stream_bytes_to_assemblyai(data):
    """Upload raw audio bytes to AssemblyAI, return upload URL."""
    print(f"Uploading {len(data)} bytes to AssemblyAI...")
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers=headers,
        data=data
    )
    if response.status_code != 200:
        raise Exception(f"AssemblyAI upload failed: {response.status_code} - {response.text}")
    upload_url = response.json()["upload_url"]
    print(f"AssemblyAI upload URL: {upload_url}")
    return upload_url


def start_transcription(audio_url):
    """Submit transcription job to AssemblyAI, return transcript_id immediately."""
    print("Submitting transcription job...")
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }
    config = {
        "audio_url": audio_url,
        "speech_models": ["universal-2"],
        "speaker_labels": True,
        "punctuate": True,
        "format_text": True,
    }
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=config,
        headers=headers
    )
    print(f"Transcription submit status: {response.status_code}")
    print(f"Transcription submit response: {response.text[:500]}")
    if response.status_code != 200:
        raise Exception(f"Transcription submit failed: {response.status_code} - {response.text}")
    data = response.json()
    if 'id' not in data:
        raise Exception(f"No 'id' in transcription response: {data}")
    transcript_id = data["id"]
    print(f"Transcription job started: {transcript_id}")
    return transcript_id


def get_transcription(transcript_id):
    """Fetch current transcription state from AssemblyAI (single request, no polling)."""
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"AssemblyAI status check failed: {response.status_code} - {response.text}")
    return response.json()


FILLER_WORDS = {
    'um', 'uh', 'hmm', 'mhm', 'hm',
    'like', 'basically', 'literally', 'actually', 'honestly',
    'right', 'okay', 'ok', 'so', 'well',
    'you know', 'i mean', 'kind of', 'sort of', 'you know what i mean',
}


def _find_fillers(words):
    """Scan word list for filler words/phrases; return list of {text, start_ms, end_ms, speaker}."""
    found = []
    i = 0
    while i < len(words):
        w = words[i]
        tok = w.get('text', '').lower().strip('.,!?;:')
        # Two-word phrases first
        if i + 1 < len(words):
            two = tok + ' ' + words[i + 1].get('text', '').lower().strip('.,!?;:')
            if two in FILLER_WORDS:
                found.append({
                    'text': two,
                    'start_ms': w.get('start', 0),
                    'end_ms': words[i + 1].get('end', 0),
                    'speaker': w.get('speaker', '?'),
                })
                i += 2
                continue
        if tok in FILLER_WORDS:
            found.append({
                'text': tok,
                'start_ms': w.get('start', 0),
                'end_ms': w.get('end', 0),
                'speaker': w.get('speaker', '?'),
            })
        i += 1
    return found


def _find_pauses(words, min_ms=1000):
    """Return list of pauses longer than min_ms between consecutive words."""
    pauses = []
    for i in range(len(words) - 1):
        gap_start = words[i].get('end', 0)
        gap_end = words[i + 1].get('start', 0)
        gap_ms = gap_end - gap_start
        if gap_ms >= min_ms:
            pauses.append({
                'start_ms': gap_start,
                'end_ms': gap_end,
                'duration_ms': gap_ms,
                'before': words[i].get('text', ''),
                'after': words[i + 1].get('text', ''),
            })
    return pauses


def _extract_json_array(text):
    """
    Return the first complete JSON array found in text using balanced-bracket
    scanning. More robust than a greedy regex when there is trailing prose.
    """
    start = text.find('[')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def analyze_transcript_with_claude(transcript_data, requirements, custom_instructions=""):
    """
    Pre-detect fillers and pauses in Python, then ask Claude only for
    confirmation / content-level editorial decisions. Keeps the prompt small
    so the request stays well within Railway's timeout.
    """
    print("Analyzing transcript with Claude...")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    words = transcript_data.get("words", [])
    utterances = transcript_data.get("utterances", [])

    # --- Build a compact utterance transcript for content context ---
    utt_lines = []
    for utt in utterances[:80]:
        start = format_timestamp(utt.get("start", 0))
        speaker = utt.get("speaker", "?")
        text = utt.get("text", "")
        utt_lines.append(f"[{start}] Speaker {speaker}: {text}")
    utt_text = "\n".join(utt_lines) if utt_lines else transcript_data.get("text", "")[:4000]

    # --- Pre-detect fillers ---
    remove_fillers = requirements.get('removeFillerWords', True)
    fillers = _find_fillers(words) if remove_fillers else []
    filler_lines = [
        f'{f["start_ms"]} {f["end_ms"]} "{f["text"]}" (Speaker {f["speaker"]})'
        for f in fillers[:200]
    ]
    filler_text = "\n".join(filler_lines) if filler_lines else "None detected"

    # --- Pre-detect long pauses ---
    remove_pauses = requirements.get('removeLongPauses', True)
    pauses = _find_pauses(words, min_ms=1500) if remove_pauses else []
    pause_lines = [
        f'{p["start_ms"]} {p["end_ms"]} {p["duration_ms"]}ms  ("{p["before"]}" → "{p["after"]}")'
        for p in pauses[:100]
    ]
    pause_text = "\n".join(pause_lines) if pause_lines else "None detected"

    print(f"Pre-detected {len(fillers)} fillers, {len(pauses)} pauses")

    prompt = f"""You are an expert podcast editor. Review the pre-detected issues and transcript below, then return a final edit decision list with precise millisecond timestamps.

UTTERANCE TRANSCRIPT (for context):
{utt_text}

PRE-DETECTED FILLER WORDS (format: start_ms end_ms "word" speaker):
{filler_text}

PRE-DETECTED PAUSES >1500ms (format: start_ms end_ms duration before→after):
{pause_text}

CLIENT REQUIREMENTS:
- Remove filler words: {remove_fillers}
- Trim long pauses to 800ms: {remove_pauses}
- Target length: {requirements.get('targetLength', 'Not specified')}

CUSTOM INSTRUCTIONS:
{custom_instructions if custom_instructions else "None"}

INSTRUCTIONS:
1. Confirm filler removals: include each filler as a "Remove Filler" decision using the EXACT start_ms and end_ms provided above.
2. Trim long pauses: for each pause, set start_ms = pause_start_ms + 800, end_ms = pause_end_ms (keeps 800ms of natural pause).
3. Add "Content Cut" decisions for any sections in the transcript that should be removed for editorial reasons (use ms values matching utterance start/end times).
4. Add "Note" decisions (no start_ms/end_ms) for observations that don't require a cut.
5. Use ONLY ms values from the data above — never invent values.

Return ONLY a JSON array, no other text. Do not wrap in code fences. Start your response with [ and end with ].
Example format:
[
  {{"type": "Remove Filler", "description": "Remove 'um'", "start_ms": 12680, "end_ms": 12900, "confidence": 95, "rationale": "Filler word"}},
  {{"type": "Trim Pause", "description": "Trim 2500ms pause to 800ms", "start_ms": 15800, "end_ms": 17500, "confidence": 90, "rationale": "Excessive pause"}},
  {{"type": "Note", "description": "Strong section, no edit needed", "confidence": 100, "rationale": "Preserve energy"}}
]"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    response_text = message.content[0].text.strip()
    # Strip code fences if present (e.g. ```json ... ```)
    if response_text.startswith('```'):
        response_text = re.sub(r'^```\w*\n?', '', response_text)
        response_text = re.sub(r'\n?```$', '', response_text).strip()
    print(f"Claude response (first 300 chars): {response_text[:300]}")

    try:
        edit_decisions = json.loads(response_text)
    except json.JSONDecodeError:
        # Balanced-bracket scan as a last resort
        array_text = _extract_json_array(response_text)
        if array_text:
            edit_decisions = json.loads(array_text)
        else:
            print(f"Full Claude response for debugging:\n{response_text}")
            raise Exception("Could not parse Claude's response as JSON")

    print(f"Generated {len(edit_decisions)} edit decisions")
    return {
        "edit_decisions": edit_decisions,
        "analysis_timestamp": datetime.now().isoformat()
    }


def apply_audio_edits(audio_path, cuts_ms):
    """
    Remove segments from audio using pydub.
    cuts_ms: list of (start_ms, end_ms) tuples to remove.
    Returns path to the edited output file.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    # Clamp, sort, and merge overlapping cuts
    cuts = sorted(
        [(max(0, int(s)), min(total_ms, int(e))) for s, e in cuts_ms if e > s],
        key=lambda x: x[0]
    )
    merged = []
    for s, e in cuts:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Build segments to keep (inverse of cuts)
    segments = []
    pos = 0
    for s, e in merged:
        if s > pos:
            segments.append(audio[pos:s])
        pos = e
    if pos < total_ms:
        segments.append(audio[pos:])

    if not segments:
        edited = audio
    else:
        edited = segments[0]
        for seg in segments[1:]:
            edited += seg

    output_path = audio_path.rsplit('.', 1)[0] + '_edited.mp3'
    edited.export(output_path, format='mp3', bitrate='192k')
    print(f"Exported edited audio to {output_path} ({len(edited)}ms)")
    return output_path


def format_timestamp(milliseconds):
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_edit_report(filename, transcript_data, edit_analysis, requirements):
    decisions = edit_analysis['edit_decisions']
    cuts = [d for d in decisions if 'start_ms' in d and 'end_ms' in d]
    report = f"""
================================================================================
                        AI PODCAST EDIT REPORT
================================================================================

EPISODE: {filename}
PROCESSED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
DURATION: {format_timestamp(transcript_data.get('audio_duration', 0))}

--------------------------------------------------------------------------------
CLIENT REQUIREMENTS
--------------------------------------------------------------------------------
Remove filler words: {requirements.get('removeFillerWords', True)}
Remove long pauses: {requirements.get('removeLongPauses', True)}
Normalize audio: {requirements.get('normalizeAudio', True)}
Remove background noise: {requirements.get('removeBackgroundNoise', True)}
Target length: {requirements.get('targetLength', 'Not specified')}

--------------------------------------------------------------------------------
TRANSCRIPT STATISTICS
--------------------------------------------------------------------------------
Total words: {len(transcript_data.get('words', []))}
Speakers detected: {len(set(u.get('speaker') for u in transcript_data.get('utterances', []) if u.get('speaker')))}
Confidence: {transcript_data.get('confidence', 0) or 0:.1%}

--------------------------------------------------------------------------------
EDIT DECISION LIST (EDL)
--------------------------------------------------------------------------------
Total decisions: {len(decisions)}
Audio cuts to apply: {len(cuts)}

"""
    for i, d in enumerate(decisions, 1):
        ts = f"{format_timestamp(d['start_ms'])} → {format_timestamp(d['end_ms'])}" if 'start_ms' in d else "N/A"
        report += f"[{i}] {ts} — {d.get('type', 'N/A')}\n"
        report += f"    {d.get('description', '')}\n"
        report += f"    Confidence: {d.get('confidence', 0)}% | {d.get('rationale', '')}\n\n"

    report += """
--------------------------------------------------------------------------------
HUMAN EDITOR CHECKLIST
--------------------------------------------------------------------------------
[ ] Listen to full edited episode
[ ] Verify no jarring cuts or audio artifacts
[ ] Confirm natural speech flow maintained
[ ] Check all client requirements met
[ ] Validate audio levels are consistent
[ ] Review all flagged sections

NOTES:
_____________________________________________________________________
_____________________________________________________________________

EDITOR SIGNATURE: _________________ DATE: _____________

================================================================================
"""
    return report


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "service": "AI Podcast Editor API",
        "version": "3.0.0"
    })


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """
    Step 1: Receive audio file, upload to AssemblyAI, start transcription.
    Also saves audio to /tmp for later editing.
    """
    if request.method == 'OPTIONS':
        return '', 204

    print("=" * 60)
    print("UPLOAD REQUEST RECEIVED")
    print("=" * 60)

    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use MP3, WAV, M4A, AAC, or OGG"}), 400

    print(f"Received file: {file.filename}")
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'mp3'

    try:
        file_data = file.read()
        print(f"Read {len(file_data)} bytes")

        upload_url = stream_bytes_to_assemblyai(file_data)
        transcript_id = start_transcription(upload_url)

        # Save to /tmp so /api/edit-audio can access it later
        tmp_path = f"/tmp/{transcript_id}.{ext}"
        with open(tmp_path, 'wb') as f:
            f.write(file_data)
        print(f"Audio saved to {tmp_path}")

        return jsonify({
            "success": True,
            "transcript_id": transcript_id,
            "filename": secure_filename(file.filename),
            "message": "File uploaded and transcription started"
        })
    except Exception as e:
        print(f"\nUPLOAD ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/transcription-status/<transcript_id>', methods=['GET'])
def transcription_status(transcript_id):
    """Return current AssemblyAI transcription status without blocking."""
    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500
    try:
        data = get_transcription(transcript_id)
        status = data.get("status")
        return jsonify({
            "status": status,
            "error": data.get("error") if status == "error" else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _run_analysis_job(job_id, transcript_data, filename, requirements, custom_instructions):
    """Background thread: run Claude analysis and store result in _jobs."""
    try:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'analyzing'

        edit_analysis = analyze_transcript_with_claude(transcript_data, requirements, custom_instructions)
        report = generate_edit_report(filename, transcript_data, edit_analysis, requirements)
        cuts_count = sum(1 for d in edit_analysis['edit_decisions'] if 'start_ms' in d and 'end_ms' in d)

        result = {
            "success": True,
            "edit_decisions": edit_analysis['edit_decisions'],
            "cuts_count": cuts_count,
            "report": report,
            "transcript": {
                "duration": transcript_data.get('audio_duration', 0),
                "words": len(transcript_data.get('words', [])),
                "speakers": len(set(u.get('speaker') for u in transcript_data.get('utterances', []) if u.get('speaker'))),
                "confidence": transcript_data.get('confidence', 0) or 0
            }
        }
        with _jobs_lock:
            _jobs[job_id] = {'status': 'completed', 'result': result}
        print(f"Job {job_id} complete: {cuts_count} cuts")

    except anthropic.APIStatusError as e:
        err = "Claude API is temporarily overloaded. Please try again." if e.status_code == 529 else str(e)
        retryable = e.status_code == 529
        with _jobs_lock:
            _jobs[job_id] = {'status': 'error', 'error': err, 'retryable': retryable}
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _jobs_lock:
            _jobs[job_id] = {'status': 'error', 'error': str(e)}


@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_podcast():
    """
    Step 2: Fetch transcript from AssemblyAI, kick off Claude analysis in a
    background thread, and return a job_id immediately.
    Poll /api/process-status/<job_id> for the result.
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        print("=" * 60)
        print("PROCESSING REQUEST RECEIVED")
        print("=" * 60)

        if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
            return jsonify({"error": "API keys not configured on server"}), 500

        data = request.json
        transcript_id = data.get('transcript_id')
        filename = data.get('filename', 'episode')
        requirements = data.get('requirements', {})
        custom_instructions = data.get('customInstructions', '')

        if not transcript_id:
            return jsonify({"error": "No transcript_id provided"}), 400

        print(f"transcript_id: {transcript_id}")

        # Fetch transcript synchronously (fast, ~1s)
        transcript_data = get_transcription(transcript_id)
        status = transcript_data.get("status")
        if status == "error":
            raise Exception(f"Transcription failed: {transcript_data.get('error', 'Unknown error')}")
        if status != "completed":
            return jsonify({"error": f"Transcription not ready (status: {status})."}), 400
        print(f"Transcript fetched. Duration: {transcript_data.get('audio_duration')}ms")

        # Kick off Claude analysis in background thread
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {'status': 'pending'}

        thread = threading.Thread(
            target=_run_analysis_job,
            args=(job_id, transcript_data, filename, requirements, custom_instructions),
            daemon=True
        )
        thread.start()
        print(f"Analysis job {job_id} started in background")

        return jsonify({"success": True, "job_id": job_id})

    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/process-status/<job_id>', methods=['GET'])
def process_status(job_id):
    """Poll this endpoint after /api/process to get Claude analysis results."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job['status'] == 'completed':
        return jsonify(job['result'])
    if job['status'] == 'error':
        return jsonify({"error": job['error'], "retryable": job.get('retryable', False)}), 500
    return jsonify({"status": job['status']})


@app.route('/api/edit-audio', methods=['POST', 'OPTIONS'])
def edit_audio():
    """
    Step 3: Apply Claude's edit decisions to the saved audio file.
    Accepts JSON: { transcript_id, cuts: [{start_ms, end_ms}, ...] }
    Returns the edited audio file as a download.
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        transcript_id = data.get('transcript_id')
        cuts = data.get('cuts', [])

        if not transcript_id:
            return jsonify({"error": "No transcript_id provided"}), 400

        # Find the saved audio file
        audio_path = None
        for ext in ALLOWED_EXTENSIONS:
            path = f"/tmp/{transcript_id}.{ext}"
            if os.path.exists(path):
                audio_path = path
                break

        if not audio_path:
            return jsonify({
                "error": "Audio file not found on server. Files are cleared on restart — please re-upload your audio."
            }), 404

        cuts_ms = [
            (c['start_ms'], c['end_ms'])
            for c in cuts
            if 'start_ms' in c and 'end_ms' in c
        ]
        print(f"Applying {len(cuts_ms)} cuts to {audio_path}")

        output_path = apply_audio_edits(audio_path, cuts_ms)

        return send_file(
            output_path,
            as_attachment=True,
            download_name='edited_podcast.mp3',
            mimetype='audio/mpeg'
        )

    except Exception as e:
        print(f"Edit audio error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
        "claude_configured": bool(CLAUDE_API_KEY),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
