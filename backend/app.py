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
from pathlib import Path

app = Flask(__name__)

# CORS: allow all origins without credentials (required for GitHub Pages -> Railway)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Get API keys from environment variables
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')

if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
    print("WARNING: API keys not set. Please set ASSEMBLYAI_API_KEY and CLAUDE_API_KEY environment variables")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_to_assemblyai(audio_file_path):
    """Upload audio file to AssemblyAI and return upload URL"""
    print(f"Uploading to AssemblyAI: {audio_file_path}")
    headers = {"authorization": ASSEMBLYAI_API_KEY}

    with open(audio_file_path, "rb") as f:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=f
        )

    if response.status_code == 200:
        upload_url = response.json()["upload_url"]
        print(f"Upload successful: {upload_url}")
        return upload_url
    else:
        raise Exception(f"Upload failed: {response.status_code} - {response.text}")


def transcribe_audio(audio_url):
    """Transcribe audio using AssemblyAI with speaker detection"""
    print("Starting transcription...")
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }

    transcription_config = {
        "audio_url": audio_url,
        "speech_model": "universal-2",
        "speaker_labels": True,
        "punctuate": True,
        "format_text": True,
    }

    print(f"Sending transcription request...")
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=transcription_config,
        headers=headers
    )

    print(f"Transcription API response status: {response.status_code}")
    print(f"Transcription API response: {response.text[:500]}")

    if response.status_code != 200:
        raise Exception(f"Transcription request failed: {response.status_code} - {response.text}")

    response_data = response.json()

    if 'id' not in response_data:
        raise Exception(f"No 'id' in response. Full response: {response_data}")

    transcript_id = response_data["id"]
    print(f"Transcription job created: {transcript_id}")

    polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    # Poll for completion
    while True:
        poll_response = requests.get(polling_url, headers={"authorization": ASSEMBLYAI_API_KEY})
        poll_data = poll_response.json()
        status = poll_data.get("status")

        if status == "completed":
            print("Transcription complete!")
            return poll_data
        elif status == "error":
            raise Exception(f"Transcription failed: {poll_data.get('error', 'Unknown error')}")

        print(f"Transcription status: {status}... waiting 5 seconds")
        time.sleep(5)


def analyze_transcript_with_claude(transcript_data, requirements, custom_instructions=""):
    """Use Claude to analyze transcript and generate intelligent edit decisions"""
    print("Analyzing transcript with Claude...")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    # Extract utterances
    utterances = transcript_data.get("utterances", [])

    # Build transcript text with timestamps (limit for token efficiency)
    transcript_text = ""
    for utt in utterances[:100]:
        speaker = utt.get("speaker", "Unknown")
        start_time = format_timestamp(utt.get("start", 0))
        text = utt.get("text", "")
        transcript_text += f"[{start_time}] Speaker {speaker}: {text}\n"

    if not transcript_text:
        # Fall back to plain text if no utterances
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
    """Convert milliseconds to HH:MM:SS format"""
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_edit_report(filename, transcript_data, edit_analysis, requirements):
    """Generate a comprehensive edit report"""
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
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "AI Podcast Editor API",
        "version": "1.0.0"
    })


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Handle file upload"""
    if request.method == 'OPTIONS':
        return '', 204

    print("=" * 60)
    print("UPLOAD REQUEST RECEIVED")
    print("=" * 60)

    if 'file' not in request.files:
        print("ERROR: No file in request")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        print("ERROR: Empty filename")
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        print(f"ERROR: File type not allowed: {file.filename}")
        return jsonify({"error": "File type not allowed. Use MP3, WAV, M4A, AAC, or OGG"}), 400

    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    print(f"File saved: {filepath}")
    print(f"File size: {os.path.getsize(filepath)} bytes")

    return jsonify({
        "success": True,
        "filename": unique_filename,
        "message": "File uploaded successfully"
    })


@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_podcast():
    """Process podcast episode with AI agents"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        print("=" * 60)
        print("PROCESSING REQUEST RECEIVED")
        print("=" * 60)

        data = request.json
        print(f"Request data keys: {list(data.keys()) if data else 'None'}")

        filename = data.get('filename')
        requirements = data.get('requirements', {})
        custom_instructions = data.get('customInstructions', '')

        print(f"Filename: {filename}")
        print(f"Requirements: {requirements}")

        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Looking for file at: {filepath}")

        if not os.path.exists(filepath):
            print(f"ERROR: File not found at {filepath}")
            return jsonify({"error": f"File not found: {filename}"}), 404

        if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
            return jsonify({"error": "API keys not configured on server"}), 500

        print("API keys verified")

        # Create job ID and output directory
        job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        # Step 1: Upload to AssemblyAI
        print("\nSTEP 1: UPLOADING TO ASSEMBLYAI")
        audio_url = upload_to_assemblyai(filepath)

        # Step 2: Transcribe
        print("\nSTEP 2: TRANSCRIBING AUDIO")
        transcript_data = transcribe_audio(audio_url)
        print(f"Transcription complete. Duration: {transcript_data.get('audio_duration')}ms")

        # Save transcript
        transcript_path = os.path.join(job_output_dir, "transcript.json")
        with open(transcript_path, "w") as f:
            json.dump(transcript_data, f, indent=2)

        # Step 3: Analyze with Claude
        print("\nSTEP 3: ANALYZING WITH CLAUDE")
        edit_analysis = analyze_transcript_with_claude(
            transcript_data,
            requirements,
            custom_instructions
        )
        print(f"Analysis complete. Generated {len(edit_analysis['edit_decisions'])} edit decisions")

        # Save edit decisions
        edit_decisions_path = os.path.join(job_output_dir, "edit_decisions.json")
        with open(edit_decisions_path, "w") as f:
            json.dump(edit_analysis, f, indent=2)

        # Step 4: Generate report
        print("\nSTEP 4: GENERATING REPORT")
        report = generate_edit_report(
            filename,
            transcript_data,
            edit_analysis,
            requirements
        )

        report_path = os.path.join(job_output_dir, "edit_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        # Clean up uploaded file
        os.remove(filepath)

        print("\nPROCESSING COMPLETE!")
        print("=" * 60)

        return jsonify({
            "success": True,
            "job_id": job_id,
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


@app.route('/api/download/<job_id>/<file_type>', methods=['GET'])
def download_file(job_id, file_type):
    """Download generated files"""
    file_mapping = {
        'transcript': 'transcript.json',
        'decisions': 'edit_decisions.json',
        'report': 'edit_report.txt'
    }

    if file_type not in file_mapping:
        return jsonify({"error": "Invalid file type"}), 400

    filepath = os.path.join(OUTPUT_FOLDER, job_id, file_mapping[file_type])

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    return send_file(filepath, as_attachment=True)


@app.route('/api/status', methods=['GET'])
def status():
    """Check API status and configuration"""
    return jsonify({
        "status": "online",
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
        "claude_configured": bool(CLAUDE_API_KEY),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
