"""
Flask Backend API for AI Podcast Editor
Handles file uploads, processes podcasts, and returns edit reports
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


def stream_to_assemblyai(file_obj):
    """Stream file object directly to AssemblyAI, return upload URL."""
    print("Streaming file to AssemblyAI...")
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers=headers,
        data=file_obj
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
        "speech_model": "universal-2",
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


def poll_transcription(transcript_id):
    """Poll AssemblyAI until transcription is complete, return transcript data."""
    print(f"Polling transcription {transcript_id}...")
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        response = requests.get(url, headers=headers)
        data = response.json()
        status = data.get("status")
        if status == "completed":
            print("Transcription complete!")
            return data
        elif status == "error":
            raise Exception(f"Transcription failed: {data.get('error', 'Unknown error')}")
        print(f"Transcription status: {status}... waiting 5s")
        time.sleep(5)


def analyze_transcript_with_claude(transcript_data, requirements, custom_instructions=""):
    """Use Claude to analyze transcript and generate intelligent edit decisions."""
    print("Analyzing transcript with Claude...")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    utterances = transcript_data.get("utterances", [])
    transcript_text = ""
    for utt in utterances[:100]:
        speaker = utt.get("speaker", "Unknown")
        start_time = format_timestamp(utt.get("start", 0))
        text = utt.get("text", "")
        transcript_text += f"[{start_time}] Speaker {speaker}: {text}\n"

    if not transcript_text:
        transcript_text = transcript_data.get("text", "")[:5000]

    prompt = f"""You are an expert podcast editor. Analyze this podcast transcript and generate intelligent editing decisions.

TRANSCRIPT:
{transcript_text}

CLIENT REQUIREMENTS:
- Remove filler words: {requirements.get('removeFillerWords', True)}
- Remove long pauses: {requirements.get('removeLongPauses', True)}
- Normalize audio: {requirements.get('normalizeAudio', True)}
- Remove background noise: {requirements.get('removeBackgroundNoise', True)}
- Target length: {requirements.get('targetLength', 'Not specified')}

CUSTOM INSTRUCTIONS:
{custom_instructions if custom_instructions else "None"}

Generate a detailed edit decision list (EDL). For each edit decision, provide:
1. Timestamp (format: HH:MM:SS)
2. Edit type (Remove Filler, Trim Pause, Keep Pause, Audio Fix, Content Cut)
3. Description of what to do
4. Confidence score (0-100)
5. Rationale for the decision

Focus on:
- Removing excessive filler words while keeping natural speech
- Trimming long pauses (>2 seconds) to 1-1.5 seconds
- Preserving intentional pauses for comedic timing
- Flagging sections that need human review
- Maintaining natural flow and energy

Return ONLY a JSON array with this structure:
[
  {{
    "timestamp": "00:02:34",
    "type": "Remove Filler",
    "description": "Remove 'um, uh' (2 instances)",
    "confidence": 95,
    "rationale": "Consecutive filler words with no content value"
  }}
]

Return ONLY the JSON array, no other text."""

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


def format_timestamp(milliseconds):
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_edit_report(filename, transcript_data, edit_analysis, requirements):
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
Total edits: {len(edit_analysis['edit_decisions'])}

"""
    for i, decision in enumerate(edit_analysis['edit_decisions'], 1):
        report += f"""
[{i}] {decision.get('timestamp', 'N/A')} - {decision.get('type', 'N/A')}
    Description: {decision.get('description', 'N/A')}
    Confidence: {decision.get('confidence', 0)}%
    Rationale: {decision.get('rationale', 'N/A')}
"""
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
        "version": "2.0.0"
    })


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """
    Step 1: Receive audio file, stream it to AssemblyAI, start transcription.
    Returns transcript_id immediately — no local file storage needed.
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

    # Stream directly to AssemblyAI — no local disk write
    upload_url = stream_to_assemblyai(file)

    # Submit transcription job (returns immediately with ID)
    transcript_id = start_transcription(upload_url)

    return jsonify({
        "success": True,
        "transcript_id": transcript_id,
        "filename": secure_filename(file.filename),
        "message": "File uploaded and transcription started"
    })


@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_podcast():
    """
    Step 2: Poll AssemblyAI for transcript, then run Claude analysis.
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
        print(f"Requirements: {requirements}")

        # Step 1: Wait for transcription to finish
        print("\nSTEP 1: POLLING TRANSCRIPTION")
        transcript_data = poll_transcription(transcript_id)
        print(f"Transcription complete. Duration: {transcript_data.get('audio_duration')}ms")

        # Step 2: Analyze with Claude
        print("\nSTEP 2: ANALYZING WITH CLAUDE")
        edit_analysis = analyze_transcript_with_claude(
            transcript_data,
            requirements,
            custom_instructions
        )
        print(f"Analysis complete. Generated {len(edit_analysis['edit_decisions'])} edit decisions")

        # Step 3: Generate report
        print("\nSTEP 3: GENERATING REPORT")
        report = generate_edit_report(filename, transcript_data, edit_analysis, requirements)

        print("\nPROCESSING COMPLETE!")
        print("=" * 60)

        return jsonify({
            "success": True,
            "edit_decisions": edit_analysis['edit_decisions'],
            "report": report,
            "transcript": {
                "duration": transcript_data.get('audio_duration', 0),
                "words": len(transcript_data.get('words', [])),
                "speakers": len(set(u.get('speaker') for u in transcript_data.get('utterances', []) if u.get('speaker'))),
                "confidence": transcript_data.get('confidence', 0) or 0
            }
        })

    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
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
