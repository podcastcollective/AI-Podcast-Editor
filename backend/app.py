"""
Flask Backend API for AI Podcast Editor
Handles file uploads, transcription, Claude analysis, and audio editing.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import time
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


def analyze_transcript_with_claude(transcript_data, requirements, custom_instructions=""):
    """Use Claude to analyze transcript and generate edit decisions with ms-precise timestamps."""
    print("Analyzing transcript with Claude...")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    words = transcript_data.get("words", [])

    # Build word-level transcript with ms timestamps (cap at 5000 words ~33 min)
    word_lines = []
    for w in words[:5000]:
        word_lines.append(
            f'{w.get("start", 0)} {w.get("end", 0)} {json.dumps(w.get("text", ""))} {w.get("speaker", "?")}'
        )
    word_text = "\n".join(word_lines)

    # Detect pauses > 1000ms between consecutive words
    pause_lines = []
    for i in range(len(words) - 1):
        gap_start = words[i].get("end", 0)
        gap_end = words[i + 1].get("start", 0)
        gap_ms = gap_end - gap_start
        if gap_ms > 1000:
            pause_lines.append(f"{gap_start} {gap_end} {gap_ms}ms")
    pause_text = "\n".join(pause_lines[:300]) if pause_lines else "None detected"

    prompt = f"""You are an expert podcast editor. Analyze the word-level transcript below and produce a list of precise audio edits.

WORD-LEVEL TRANSCRIPT (format: start_ms end_ms "word" speaker):
{word_text}

DETECTED PAUSES >1000ms (format: start_ms end_ms duration):
{pause_text}

CLIENT REQUIREMENTS:
- Remove filler words (um, uh, like, you know, basically, right, etc.): {requirements.get('removeFillerWords', True)}
- Trim long pauses to 800ms: {requirements.get('removeLongPauses', True)}
- Target length: {requirements.get('targetLength', 'Not specified')}

CUSTOM INSTRUCTIONS:
{custom_instructions if custom_instructions else "None"}

RULES:
1. For filler words: use the word's exact start_ms and end_ms from the transcript.
2. For long pauses to trim: set start_ms = pause_start_ms + 800, end_ms = pause_end_ms (keeps first 800ms).
3. For content cuts: span the full ms range of the words to remove.
4. Decisions without a physical cut (notes, keep decisions) must omit start_ms and end_ms entirely.
5. Use ONLY millisecond values that appear in the transcript above — do not invent values.

Return ONLY a JSON array, no other text:
[
  {{
    "type": "Remove Filler",
    "description": "Remove 'um'",
    "start_ms": 12680,
    "end_ms": 12900,
    "confidence": 95,
    "rationale": "Filler word with no content value"
  }},
  {{
    "type": "Trim Pause",
    "description": "Trim 2100ms pause to 800ms",
    "start_ms": 15700,
    "end_ms": 17000,
    "confidence": 90,
    "rationale": "Excessive pause after sentence"
  }},
  {{
    "type": "Note",
    "description": "Good energy in this section, no edit needed",
    "confidence": 100,
    "rationale": "Preserve natural delivery"
  }}
]"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text

    try:
        edit_decisions = json.loads(response_text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            edit_decisions = json.loads(json_match.group())
        else:
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


@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_podcast():
    """
    Step 2: Fetch completed transcript from AssemblyAI, run Claude analysis.
    Accepts JSON: { transcript_id, filename, requirements, customInstructions }
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

        # Fetch transcript (client already polled until complete)
        print("\nSTEP 1: FETCHING TRANSCRIPT")
        transcript_data = get_transcription(transcript_id)
        status = transcript_data.get("status")
        if status == "error":
            raise Exception(f"Transcription failed: {transcript_data.get('error', 'Unknown error')}")
        if status != "completed":
            return jsonify({"error": f"Transcription not ready (status: {status}). Poll /api/transcription-status first."}), 400
        print(f"Transcription complete. Duration: {transcript_data.get('audio_duration')}ms")

        # Analyze with Claude
        print("\nSTEP 2: ANALYZING WITH CLAUDE")
        edit_analysis = analyze_transcript_with_claude(
            transcript_data,
            requirements,
            custom_instructions
        )
        print(f"Analysis complete. Generated {len(edit_analysis['edit_decisions'])} edit decisions")

        # Generate report
        print("\nSTEP 3: GENERATING REPORT")
        report = generate_edit_report(filename, transcript_data, edit_analysis, requirements)

        print("\nPROCESSING COMPLETE!")
        print("=" * 60)

        cuts_count = sum(1 for d in edit_analysis['edit_decisions'] if 'start_ms' in d and 'end_ms' in d)
        return jsonify({
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
        })

    except anthropic.APIStatusError as e:
        if e.status_code == 529:
            print(f"\nClaude overloaded (529): {e}")
            return jsonify({"error": "Claude API is temporarily overloaded. Please try again in a moment.", "retryable": True}), 503
        print(f"\nClaude API error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
