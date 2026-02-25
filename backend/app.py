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
AUPHONIC_API_KEY = os.environ.get('AUPHONIC_API_KEY')    # Bearer token from auphonic.com/accounts/api-access/
CLEANVOICE_API_KEY = os.environ.get('CLEANVOICE_API_KEY')  # API key from cleanvoice.ai/dashboard
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')    # API key from elevenlabs.io/developers

if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
    print("WARNING: API keys not set. Please set ASSEMBLYAI_API_KEY and CLAUDE_API_KEY environment variables")

# In-memory job stores (Claude analysis + audio editing).
# Gunicorn must use threads (not multiple processes) so these dicts are shared.
_jobs: dict = {}
_jobs_lock = threading.Lock()
_edit_jobs: dict = {}
_edit_jobs_lock = threading.Lock()


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
    'um', 'uh', 'uhm', 'hmm', 'mhm', 'hm', 'mm', 'ah', 'eh',
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
    pauses = _find_pauses(words, min_ms=2000) if remove_pauses else []
    pause_lines = [
        f'{p["start_ms"]} {p["end_ms"]} {p["duration_ms"]}ms  ("{p["before"]}" → "{p["after"]}")'
        for p in pauses[:100]
    ]
    pause_text = "\n".join(pause_lines) if pause_lines else "None detected"

    print(f"Pre-detected {len(fillers)} fillers, {len(pauses)} pauses")

    prompt = f"""You are an experienced podcast editor. Your task is to clean and polish this podcast episode while keeping a natural, human flow. The goal is clean, confident, and human — NOT over-edited.

UTTERANCE TRANSCRIPT (for context):
{utt_text}

PRE-DETECTED FILLER WORDS (format: start_ms end_ms "word" speaker):
{filler_text}

PRE-DETECTED PAUSES >2s (format: start_ms end_ms duration before→after):
{pause_text}

CLIENT REQUIREMENTS:
- Remove filler words: {remove_fillers}
- Trim long pauses: {remove_pauses}
- Target length: {requirements.get('targetLength', 'Not specified')}

CUSTOM INSTRUCTIONS:
{custom_instructions if custom_instructions else "None"}

EDITING PHILOSOPHY:
- Preserve personality, rhythm, and emotion. The episode should sound human, not robotic.
- Avoid abrupt or robotic cuts — edits must sound conversational and smooth.
- Do NOT over-edit. When in doubt, leave it in.

FILLER WORD RULES:
- Remove approximately 50% of the filler words listed above — NOT all of them.
- Keep fillers that serve as natural transitions or thinking pauses.
- Remove fillers that cluster together or interrupt the flow of a clear thought.
- Use the EXACT start_ms and end_ms provided. CRITICAL: Never adjust these timestamps — they are word-level boundaries from the transcription engine.
- Preserve ALL acronyms and industry-specific terms exactly as spoken — they are key terminology, not mistakes.

PAUSE RULES:
- For pauses listed above, trim so that any pause longer than 2 seconds becomes about 1 second.
- Set start_ms = pause_start_ms + 1000, end_ms = pause_end_ms (keeps ~1s of natural pause).
- Do NOT remove short, intentional pauses used for emphasis — only trim the clearly excessive ones.

CONTENT RULES:
- Look for FALSE STARTS where a speaker starts a sentence, stops, and restarts. Keep the best full version, remove the false start. Only fix stumbles and restarts that reduce clarity — leave minor ones that sound natural.
- If the same point is made twice in nearly identical ways, remove the weaker version. Cut at sentence boundaries.
- Do NOT rewrite or paraphrase content. Do NOT change the meaning or tone of the speaker.
- All content cuts MUST start at the beginning of a word (use the word's start_ms) and end at the end of a word (use the word's end_ms). NEVER cut mid-word.

STRUCTURAL RULES:
- If there is pre-interview chat before the official episode begins, mark it for removal.
- If there is post-interview chat after the episode has clearly concluded, mark it for removal.
- Preserve the full interview content itself.

OUTPUT RULES:
- Add at least one "Note" decision (no start_ms/end_ms) summarizing the overall edit.
- Use ONLY ms values from the data above — never invent values.
- You MUST include at least one decision.

Call the submit_edit_decisions tool with your decisions.

Example tool input for reference:
{{"decisions": [{{"type": "Remove Filler", "description": "Remove 'um'", "start_ms": 12680, "end_ms": 12900, "confidence": 95, "rationale": "Filler word — interrupts flow"}}, {{"type": "Trim Pause", "description": "Trim 3200ms pause to ~1s", "start_ms": 15000, "end_ms": 17200, "confidence": 90, "rationale": "Excessive pause"}}, {{"type": "Content Cut", "description": "Remove false start", "start_ms": 22000, "end_ms": 23500, "confidence": 85, "rationale": "Speaker restarts sentence more clearly"}}, {{"type": "Note", "description": "Light edit — preserved natural conversational flow", "confidence": 100, "rationale": "Summary"}}]}}"""

    # Use tool_use to force structured JSON output — no text parsing needed.
    tools = [{
        "name": "submit_edit_decisions",
        "description": "Submit the final list of edit decisions for the podcast episode.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "description": "One of: Remove Filler, Trim Pause, Content Cut, Note"},
                            "description": {"type": "string"},
                            "start_ms": {"type": "integer"},
                            "end_ms": {"type": "integer"},
                            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                            "rationale": {"type": "string"}
                        },
                        "required": ["type", "description", "confidence", "rationale"]
                    }
                }
            },
            "required": ["decisions"]
        }
    }]

    # Retry with exponential backoff for transient API errors (overloaded, rate limit)
    # Overload can persist for minutes — retry aggressively with long waits
    import time as _time
    max_retries = 8
    message = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_edit_decisions"},
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            break
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                wait = min(10 * (2 ** attempt), 120)  # 10s, 20s, 40s, 80s, 120s, 120s, 120s
                print(f"Claude API {e.status_code}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                _time.sleep(wait)
            else:
                raise
    if message is None:
        raise Exception(f"Claude API failed after {max_retries} attempts")

    # Extract structured data from the tool call — guaranteed valid JSON.
    edit_decisions = None
    for block in message.content:
        if block.type == "tool_use" and block.name == "submit_edit_decisions":
            edit_decisions = block.input.get("decisions", [])
            break

    if edit_decisions is None:
        print(f"Unexpected response blocks: {[b.type for b in message.content]}")
        raise Exception("Claude did not return tool call — unexpected response format")

    print(f"Generated {len(edit_decisions)} edit decisions")
    return {
        "edit_decisions": edit_decisions,
        "analysis_timestamp": datetime.now().isoformat()
    }





def _snap_to_zero_crossing(audio, ms, search_radius_ms=5):
    """
    Snap to the nearest zero-crossing within ±search_radius_ms.
    Tiny radius — just eliminates waveform pops, never moves far enough
    to clip into adjacent words.
    """
    samples = audio.get_array_of_samples()
    sample_rate = audio.frame_rate
    channels = audio.channels
    center_sample = int(ms * sample_rate / 1000) * channels
    radius_samples = int(search_radius_ms * sample_rate / 1000) * channels
    s_lo = max(0, center_sample - radius_samples)
    s_hi = min(len(samples) - channels, center_sample + radius_samples)
    best_idx = center_sample
    best_val = abs(samples[min(center_sample, len(samples) - 1)])
    for i in range(s_lo, s_hi, channels):
        val = abs(samples[i])
        if val < best_val:
            best_val = val
            best_idx = i
    return int(best_idx / channels * 1000 / sample_rate)


def apply_audio_edits(audio_path, cuts_ms, words=None):
    """
    Remove segments from audio using pydub, apply noise gate, export as WAV.
    cuts_ms: list of (start_ms, end_ms) tuples to remove.
    words: optional list of word dicts with 'start' and 'end' ms from transcript.
           Used to snap cut boundaries to word edges so we never clip mid-word.
    Returns path to the edited WAV file.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    # Build word intervals for safety checking: [(start, end), ...]
    word_intervals = [(w.get('start', 0), w.get('end', 0)) for w in (words or []) if w.get('end', 0) > w.get('start', 0)]

    def _lands_inside_word(ms):
        """Check if a timestamp falls inside any word's audio."""
        for ws, we in word_intervals:
            if ws < ms < we:
                return (ws, we)
        return None

    def _safe_cut_start(ms):
        """Ensure cut start doesn't land inside a word.
        If it does, move it backward to that word's start (removing the
        whole word rather than clipping it)."""
        hit = _lands_inside_word(ms)
        if hit:
            print(f"  Cut start {ms}ms was inside word [{hit[0]}-{hit[1]}], moved to {hit[0]}ms")
            return hit[0]
        return ms

    def _safe_cut_end(ms):
        """Ensure cut end doesn't land inside a word.
        If it does, move it forward to that word's end (removing the
        whole word rather than clipping it)."""
        hit = _lands_inside_word(ms)
        if hit:
            print(f"  Cut end {ms}ms was inside word [{hit[0]}-{hit[1]}], moved to {hit[1]}ms")
            return hit[1]
        return ms

    # Step 1: Clamp and sort cuts
    adjusted = []
    for s, e in cuts_ms:
        s, e = int(s), int(e)
        if e <= s:
            continue
        adjusted.append((max(0, s), min(total_ms, e)))

    # Step 2: Sort, merge overlapping cuts
    adjusted.sort(key=lambda x: x[0])
    merged = []
    for s, e in adjusted:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Step 3: Safety check — if a boundary lands mid-word, push it out
    if word_intervals:
        for cut in merged:
            cut[0] = _safe_cut_start(cut[0])
            cut[1] = _safe_cut_end(cut[1])

    # Step 4: Snap to zero-crossing (±5ms only — prevents pops, can't clip words)
    for cut in merged:
        cut[0] = _snap_to_zero_crossing(audio, cut[0])
        cut[1] = _snap_to_zero_crossing(audio, cut[1])

    # Build segments to keep — no artificial gaps inserted.
    # Natural silence already exists around filler words in the original audio;
    # adding extra gaps makes transitions sound unnatural and too slow.
    CROSSFADE_MS = 50  # overlap crossfade — smooth and inaudible for speech

    segments = []
    pos = 0
    for s, e in merged:
        if s > pos:
            segments.append(audio[pos:s])
        pos = e
    if pos < total_ms:
        segments.append(audio[pos:])

    # Join with overlap crossfade — prevents volume dips at cut points
    if segments:
        edited = segments[0]
        for seg in segments[1:]:
            # Only crossfade if both segments are long enough
            if len(edited) > CROSSFADE_MS and len(seg) > CROSSFADE_MS:
                edited = edited.append(seg, crossfade=CROSSFADE_MS)
            else:
                edited += seg
    else:
        edited = AudioSegment.empty()

    removed_ms = total_ms - len(edited)
    print(f"Cuts complete: {len(merged)} cuts, removed {removed_ms}ms, {len(edited)}ms remaining")

    # Export as WAV — Dolby.io Enhance will handle noise reduction and voice processing
    wav_path = audio_path.rsplit('.', 1)[0] + '_edited.wav'
    edited.export(wav_path, format='wav')
    print(f"Exported edited audio: {wav_path} ({len(edited)}ms, {os.path.getsize(wav_path) // 1024}KB)")
    return wav_path


def process_with_elevenlabs(wav_path):
    """
    Send audio to ElevenLabs Voice Isolator for ML-based speech enhancement.
    Removes background noise, isolates speech — similar to Adobe Enhanced Speech.
    Single synchronous POST — no polling needed.
    Returns path to the enhanced audio file.
    """
    file_size = os.path.getsize(wav_path)
    print(f"ElevenLabs: uploading {file_size // 1024 // 1024}MB for voice isolation")

    with open(wav_path, 'rb') as f:
        resp = requests.post(
            'https://api.elevenlabs.io/v1/audio-isolation',
            headers={'xi-api-key': ELEVENLABS_API_KEY},
            files={'audio': (os.path.basename(wav_path), f, 'audio/wav')},
            timeout=600,  # large files may take a while
        )

    if resp.status_code != 200:
        raise Exception(f"ElevenLabs voice isolation failed: {resp.status_code} {resp.text[:300]}")

    # Response is the enhanced audio as binary (MP3)
    output_path = wav_path.rsplit('.', 1)[0] + '_enhanced.mp3'
    with open(output_path, 'wb') as out:
        out.write(resp.content)
    print(f"ElevenLabs: enhanced audio saved: {output_path} ({len(resp.content) // 1024}KB)")
    return output_path


def _high_freq_rolloff(audio):
    """
    Gentle high-frequency rolloff to match Adobe Enhanced Speech's tonal profile.
    ElevenLabs preserves 5kHz+ content that Adobe cuts aggressively.
    A 2nd-order Butterworth LPF at 8kHz gives ~-3dB@8kHz, ~-12dB/oct above —
    closing the 3-6dB gap measured in the 5-16kHz range.
    """
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    sample_rate = audio.frame_rate
    channels = audio.channels

    if channels > 1:
        samples = samples.reshape(-1, channels)

    lpf = butter(2, 8000, btype='low', fs=sample_rate, output='sos')

    if channels > 1:
        for ch in range(channels):
            samples[:, ch] = sosfiltfilt(lpf, samples[:, ch])
        samples = samples.flatten()
    else:
        samples = sosfiltfilt(lpf, samples)

    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    from pydub import AudioSegment
    result = AudioSegment(
        data=samples.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels,
    )
    print(f"High-freq rolloff (LPF 8kHz): dBFS {audio.dBFS:.1f} -> {result.dBFS:.1f}")
    return result


def process_with_auphonic(wav_path):
    """
    Send a WAV file to Auphonic for professional audio post-production:
    loudness normalisation (-16 LUFS podcast standard) + noise reduction.
    Returns the path to the downloaded MP3 output file.
    Raises an exception on any failure so the caller can fall back to the WAV.
    """
    headers = {"Authorization": f"Bearer {AUPHONIC_API_KEY}"}

    print(f"Uploading to Auphonic: {wav_path} ({os.path.getsize(wav_path) // 1024 // 1024}MB)")
    with open(wav_path, 'rb') as f:
        resp = requests.post(
            'https://auphonic.com/api/simple/productions.json',
            headers=headers,
            files={'input_file': (os.path.basename(wav_path), f, 'audio/wav')},
            data={
                'action': 'start',
                'title': 'Podcast Edit',
                'output_basename': 'edited_podcast',
                'loudnesstarget': '-16',
                'denoise': 'true',
                'filtering': 'true',
                'normloudness': 'true',
            },
            timeout=180,
        )

    if resp.status_code not in (200, 201):
        raise Exception(f"Auphonic upload failed: {resp.status_code} — {resp.text[:300]}")

    production = resp.json().get('data', {})
    prod_uuid = production.get('uuid')
    if not prod_uuid:
        raise Exception(f"No UUID in Auphonic response: {resp.text[:300]}")
    print(f"Auphonic production started: {prod_uuid}")

    # Poll until Done (max 12 minutes, 15s interval)
    deadline = time.time() + 720
    while time.time() < deadline:
        time.sleep(15)
        check = requests.get(
            f'https://auphonic.com/api/production/{prod_uuid}.json',
            headers=headers,
            timeout=30,
        )
        if check.status_code != 200:
            raise Exception(f"Auphonic status check failed: {check.status_code}")
        data = check.json().get('data', {})
        status = data.get('status_string', '')
        print(f"Auphonic status: {status}")

        if status == 'Done':
            output_files = data.get('output_files', [])
            if not output_files:
                raise Exception("Auphonic returned no output files")
            download_url = output_files[0].get('download_url')
            if not download_url:
                raise Exception("Auphonic output has no download_url")

            print(f"Downloading Auphonic output: {download_url}")
            dl = requests.get(download_url, headers=headers, timeout=300)
            if dl.status_code != 200:
                raise Exception(f"Auphonic download failed: {dl.status_code}")

            # Detect format from the output file info or URL
            out_format = output_files[0].get('format', 'wav')
            out_ext = 'mp3' if 'mp3' in out_format.lower() else out_format.lower()
            out_path = wav_path.rsplit('.', 1)[0] + f'_auphonic.{out_ext}'
            with open(out_path, 'wb') as out:
                out.write(dl.content)
            print(f"Auphonic output saved: {out_path} ({len(dl.content) // 1024}KB)")
            return out_path

        if status in ('Error', 'Failed'):
            msg = data.get('error_message') or data.get('warning_message') or 'Unknown Auphonic error'
            raise Exception(f"Auphonic processing failed: {msg}")

    raise Exception("Auphonic timed out after 12 minutes")


def process_with_cleanvoice(audio_path):
    """
    Send audio to Cleanvoice (v2 API) for AI-powered removal of mouth noises,
    stutters, breathing, and remaining filler words at the audio level.
    Flow: get signed URL → upload file → create edit → poll → download.
    Returns path to the cleaned audio file.
    """
    headers = {"X-API-Key": CLEANVOICE_API_KEY}
    filename = os.path.basename(audio_path)
    file_size_mb = os.path.getsize(audio_path) / 1024 / 1024

    # Step 1: Get a signed upload URL
    print(f"Cleanvoice: requesting signed URL for {filename} ({file_size_mb:.1f}MB)")
    sign_resp = requests.post(
        f'https://api.cleanvoice.ai/v2/upload?filename={filename}',
        headers=headers,
        timeout=30,
    )
    if sign_resp.status_code not in (200, 201):
        raise Exception(f"Cleanvoice signed URL request failed: {sign_resp.status_code} — {sign_resp.text[:300]}")

    signed_url = sign_resp.json().get('signedUrl')
    if not signed_url:
        raise Exception(f"No signedUrl in Cleanvoice response: {sign_resp.text[:300]}")
    print(f"Cleanvoice: got signed URL")

    # Step 2: Upload the file to the signed URL
    print(f"Cleanvoice: uploading file...")
    with open(audio_path, 'rb') as f:
        put_resp = requests.put(
            signed_url,
            data=f,
            headers={'Content-Type': 'audio/wav'},
            timeout=300,
        )
    if put_resp.status_code not in (200, 201):
        raise Exception(f"Cleanvoice file upload failed: {put_resp.status_code} — {put_resp.text[:300]}")
    print(f"Cleanvoice: file uploaded")

    # Step 3: Create an edit job using the signed URL as the file reference
    edit_resp = requests.post(
        'https://api.cleanvoice.ai/v2/edits',
        headers={**headers, 'Content-Type': 'application/json'},
        json={
            "input": {
                "files": [signed_url.split('?')[0]],  # URL without query params
                "config": {
                    "remove_noise": True,
                    "fillers": True,
                    "long_silences": False,  # We already handle pauses ourselves
                }
            }
        },
        timeout=30,
    )
    if edit_resp.status_code not in (200, 201):
        raise Exception(f"Cleanvoice edit creation failed: {edit_resp.status_code} — {edit_resp.text[:300]}")

    edit_id = edit_resp.json().get('id')
    if not edit_id:
        raise Exception(f"No edit ID in Cleanvoice response: {edit_resp.text[:300]}")
    print(f"Cleanvoice edit started: {edit_id}")

    # Step 4: Poll until complete (max 10 minutes, 10s interval)
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(10)
        check = requests.get(
            f'https://api.cleanvoice.ai/v2/edits/{edit_id}',
            headers=headers,
            timeout=30,
        )
        if check.status_code != 200:
            raise Exception(f"Cleanvoice status check failed: {check.status_code} — {check.text[:300]}")
        data = check.json()
        status = data.get('status', '')
        print(f"Cleanvoice status: {status}")

        if status in ('completed', 'SUCCESS', 'done'):
            # Find the download URL — Cleanvoice v2 puts it at result.download_url
            result = data.get('result', {})
            download_url = (
                result.get('download_url') or
                result.get('url') or
                data.get('download_url') or
                data.get('output', {}).get('url')
            )
            if not download_url:
                # Try output files array
                outputs = data.get('output', {}).get('files', [])
                if outputs:
                    download_url = outputs[0] if isinstance(outputs[0], str) else outputs[0].get('url')
            if not download_url:
                raise Exception(f"Cleanvoice completed but no download URL found: {json.dumps(data)[:500]}")

            print(f"Downloading Cleanvoice output: {download_url}")
            dl = requests.get(download_url, timeout=300)
            if dl.status_code != 200:
                raise Exception(f"Cleanvoice download failed: {dl.status_code}")
            output_path = audio_path.rsplit('.', 1)[0] + '_cleanvoice.wav'
            with open(output_path, 'wb') as out:
                out.write(dl.content)
            print(f"Cleanvoice output saved: {output_path} ({len(dl.content) // 1024}KB)")
            return output_path

        if status in ('ERROR', 'error', 'failed'):
            raise Exception(f"Cleanvoice processing failed: {json.dumps(data)[:500]}")

    raise Exception("Cleanvoice timed out after 10 minutes")


def _run_edit_job(job_id, audio_path, cuts_ms, use_auphonic, transcript_id=None):
    """
    Background thread: cutting (with crossfade) → elevenlabs → high-freq rolloff →
    cleanvoice → auphonic (or loudness norm) → completed.
    Every stage is mandatory if its API key is configured — any failure stops
    the pipeline and reports the error so nothing incomplete reaches the client.
    """
    active_stage = 'init'
    try:
        # Fetch word-level timestamps from AssemblyAI for smart cut snapping
        words = None
        if transcript_id:
            try:
                transcript_data = get_transcription(transcript_id)
                words = transcript_data.get('words', [])
                print(f"Fetched {len(words)} word timestamps for cut snapping")
            except Exception as e:
                print(f"Warning: could not fetch word timestamps: {e}")

        # Stage 1: apply timestamp cuts
        active_stage = 'cutting'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'cutting'
        wav_path = apply_audio_edits(audio_path, cuts_ms, words=words)

        # Stage 2: ElevenLabs Voice Isolator — ML-based noise removal, speech isolation
        if ELEVENLABS_API_KEY:
            active_stage = 'elevenlabs'
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'elevenlabs'
            wav_path = process_with_elevenlabs(wav_path)

            # Stage 2b: Roll off highs to match Adobe Enhanced Speech tonal profile
            # ElevenLabs preserves 5kHz+ content that sounds harsh vs Adobe's rolloff
            from pydub import AudioSegment as _AS
            enhanced = _AS.from_file(wav_path)
            rolled = _high_freq_rolloff(enhanced)
            rolled_path = wav_path.rsplit('.', 1)[0] + '_rolled.wav'
            rolled.export(rolled_path, format='wav')
            wav_path = rolled_path

        # Stage 3: Cleanvoice — AI voice cleaning (filler sounds, mouth clicks)
        if CLEANVOICE_API_KEY:
            active_stage = 'cleanvoice'
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'cleanvoice'
            wav_path = process_with_cleanvoice(wav_path)

        # Stage 4: Auphonic — loudness normalisation + final mastering
        if use_auphonic and AUPHONIC_API_KEY:
            active_stage = 'auphonic'
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'auphonic'
            final_path = process_with_auphonic(wav_path)
        else:
            # Simple loudness normalization when Auphonic is not configured
            # Normalize to -16 LUFS (podcast standard) with -1dBFS ceiling
            active_stage = 'loudness_norm'
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'loudness_norm'
            from pydub import AudioSegment as _AS
            audio = _AS.from_file(wav_path)
            target_dBFS = -16.0
            change_dB = target_dBFS - audio.dBFS
            # Apply gain but cap at -1dBFS ceiling
            audio = audio.apply_gain(change_dB)
            ceiling_dBFS = -1.0
            if audio.max_dBFS > ceiling_dBFS:
                audio = audio.apply_gain(ceiling_dBFS - audio.max_dBFS)
            norm_path = wav_path.rsplit('.', 1)[0] + '_normalized.mp3'
            audio.export(norm_path, format='mp3', bitrate='192k')
            print(f"Loudness normalized to {target_dBFS} LUFS: {norm_path}")
            final_path = norm_path

        with _edit_jobs_lock:
            _edit_jobs[job_id] = {
                'status': 'completed',
                'path': final_path,
                'is_mp3': final_path.endswith('.mp3'),
            }
        print(f"Edit job {job_id} complete: {final_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with _edit_jobs_lock:
            _edit_jobs[job_id] = {
                'status': 'error',
                'error': str(e),
                'failed_step': active_stage,
            }


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
    Step 3: Start async audio editing job.
    Accepts JSON: { transcript_id, cuts: [{start_ms, end_ms}, ...] }
    Returns { job_id } immediately — poll /api/edit-audio-status/<job_id>.
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        transcript_id = data.get('transcript_id')
        cuts = data.get('cuts', [])

        if not transcript_id:
            return jsonify({"error": "No transcript_id provided"}), 400

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
        print(f"Starting edit job: {len(cuts_ms)} cuts, cleanvoice={'yes' if CLEANVOICE_API_KEY else 'no'}, auphonic={'yes' if AUPHONIC_API_KEY else 'no'}")

        job_id = str(uuid.uuid4())
        with _edit_jobs_lock:
            _edit_jobs[job_id] = {'status': 'pending'}

        thread = threading.Thread(
            target=_run_edit_job,
            args=(job_id, audio_path, cuts_ms, bool(AUPHONIC_API_KEY), transcript_id),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "cleanvoice": bool(CLEANVOICE_API_KEY),
            "auphonic": bool(AUPHONIC_API_KEY),
        })

    except Exception as e:
        print(f"Edit audio error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/edit-audio-status/<job_id>', methods=['GET'])
def edit_audio_status(job_id):
    """Poll after /api/edit-audio. Returns status or signals ready-to-download."""
    with _edit_jobs_lock:
        job = _edit_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job['status'] == 'completed':
        return jsonify({"status": "completed", "is_mp3": job.get('is_mp3', False)})
    if job['status'] == 'error':
        return jsonify({"error": job['error'], "failed_step": job.get('failed_step')}), 500
    return jsonify({"status": job['status']})


@app.route('/api/edit-audio-download/<job_id>', methods=['GET'])
def edit_audio_download(job_id):
    """Download the completed edited audio file."""
    with _edit_jobs_lock:
        job = _edit_jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({"error": "File not ready"}), 404

    path = job['path']
    is_mp3 = job.get('is_mp3', False)
    mimetype = 'audio/mpeg' if is_mp3 else 'audio/wav'
    download_name = 'edited_podcast.mp3' if is_mp3 else 'edited_podcast.wav'

    return send_file(path, as_attachment=True, download_name=download_name, mimetype=mimetype)


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
        "claude_configured": bool(CLAUDE_API_KEY),
        "cleanvoice_configured": bool(CLEANVOICE_API_KEY),
        "auphonic_configured": bool(AUPHONIC_API_KEY),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
