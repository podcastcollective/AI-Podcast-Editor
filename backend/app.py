"""
Flask Backend API for AI Podcast Editor
Handles file uploads, transcription, Claude analysis, and audio editing.
"""

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import json
import time
import threading
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import anthropic
import requests

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}

# API keys
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
CLEANVOICE_API_KEY = os.environ.get('CLEANVOICE_API_KEY')     # cleanvoice.ai/dashboard

if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY and CLAUDE_API_KEY are required")
if not CLEANVOICE_API_KEY:
    print("WARNING: CLEANVOICE_API_KEY not set — audio enhancement will be skipped")

# In-memory job stores. Gunicorn must use threads (not processes) so these are shared.
_jobs: dict = {}
_jobs_lock = threading.Lock()
_edit_jobs: dict = {}
_edit_jobs_lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# ASSEMBLYAI — transcription
# ============================================================================

def stream_bytes_to_assemblyai(data):
    """Upload raw audio bytes to AssemblyAI, return upload URL."""
    print(f"Uploading {len(data)} bytes to AssemblyAI...")
    resp = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers={"authorization": ASSEMBLYAI_API_KEY},
        data=data,
    )
    if resp.status_code != 200:
        raise Exception(f"AssemblyAI upload failed: {resp.status_code} - {resp.text}")
    upload_url = resp.json()["upload_url"]
    print(f"AssemblyAI upload URL: {upload_url}")
    return upload_url


def start_transcription(audio_url):
    """Submit transcription job to AssemblyAI, return transcript_id."""
    print("Submitting transcription job...")
    resp = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json={
            "audio_url": audio_url,
            "speech_models": ["universal-2"],
            "speaker_labels": True,
            "punctuate": True,
            "format_text": True,
        },
        headers={
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json",
        },
    )
    if resp.status_code != 200:
        raise Exception(f"Transcription submit failed: {resp.status_code} - {resp.text}")
    data = resp.json()
    if 'id' not in data:
        raise Exception(f"No 'id' in transcription response: {data}")
    print(f"Transcription job started: {data['id']}")
    return data['id']


def get_transcription(transcript_id):
    """Fetch current transcription state from AssemblyAI."""
    resp = requests.get(
        f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
        headers={"authorization": ASSEMBLYAI_API_KEY},
    )
    if resp.status_code != 200:
        raise Exception(f"AssemblyAI status check failed: {resp.status_code} - {resp.text}")
    return resp.json()


# ============================================================================
# PRESETS — control both Claude editing and Cleanvoice enhancement
# ============================================================================

PRESETS = {
    'studio': {
        'remove_fillers': True,
        'filler_pct': 40,           # remove fewer fillers — studio audio is clean
        'remove_pauses': True,
        'pause_min_ms': 2500,       # only trim very long pauses
        'studio_sound': False,      # skip heavy enhancement — already clean
        'remove_noise': False,
        'claude_hint': 'This is a studio recording with clean audio. Be very conservative with cuts — only remove clear disfluencies. The audio quality is already good, so preserve the natural sound.',
    },
    'zoom': {
        'remove_fillers': True,
        'filler_pct': 60,
        'remove_pauses': True,
        'pause_min_ms': 2000,
        'studio_sound': 'nightly',  # full dereverb + enhancement
        'remove_noise': True,
        'claude_hint': 'This is a remote/Zoom recording. Standard editing — remove clear filler words and trim long pauses while keeping conversational flow.',
    },
    'solo': {
        'remove_fillers': True,
        'filler_pct': 80,           # aggressive filler removal for narration
        'remove_pauses': True,
        'pause_min_ms': 1500,       # tighter pause trimming
        'studio_sound': 'nightly',
        'remove_noise': True,
        'claude_hint': 'This is solo narration. Be more aggressive with filler word removal since there is no conversation to preserve. Tighten pauses for a polished delivery.',
    },
    'raw': {
        'remove_fillers': True,
        'filler_pct': 80,
        'remove_pauses': True,
        'pause_min_ms': 1500,
        'studio_sound': 'nightly',
        'remove_noise': True,
        'claude_hint': 'This is a rough recording that needs heavy cleanup. Be aggressive with filler removal and pause trimming. Look for false starts and repeated sentences to cut.',
    },
}


def _get_preset(name):
    """Return preset config, defaulting to 'zoom' for unknown values."""
    return PRESETS.get(name, PRESETS['zoom'])


# ============================================================================
# CLAUDE — edit analysis
# ============================================================================

FILLER_WORDS = {
    'um', 'uh', 'uhm', 'hmm', 'mhm', 'hm', 'mm', 'ah', 'eh',
}


def _find_fillers(words):
    """Scan word list for filler words; return list of {text, start_ms, end_ms, speaker}."""
    found = []
    i = 0
    while i < len(words):
        w = words[i]
        tok = w.get('text', '').lower().strip('.,!?;:')
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


def format_timestamp(milliseconds):
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def analyze_transcript_with_claude(transcript_data, preset_cfg, custom_instructions=""):
    """Pre-detect fillers/pauses, then ask Claude for editorial decisions."""
    print("Analyzing transcript with Claude...")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    words = transcript_data.get("words", [])
    utterances = transcript_data.get("utterances", [])

    utt_lines = []
    for utt in utterances[:80]:
        start = format_timestamp(utt.get("start", 0))
        speaker = utt.get("speaker", "?")
        text = utt.get("text", "")
        utt_lines.append(f"[{start}] Speaker {speaker}: {text}")
    utt_text = "\n".join(utt_lines) if utt_lines else transcript_data.get("text", "")[:4000]

    remove_fillers = preset_cfg.get('remove_fillers', True)
    filler_pct = preset_cfg.get('filler_pct', 60)
    fillers = _find_fillers(words) if remove_fillers else []
    filler_lines = [
        f'{f["start_ms"]} {f["end_ms"]} "{f["text"]}" (Speaker {f["speaker"]})'
        for f in fillers[:200]
    ]
    filler_text = "\n".join(filler_lines) if filler_lines else "None detected"

    remove_pauses = preset_cfg.get('remove_pauses', True)
    pause_min_ms = preset_cfg.get('pause_min_ms', 2000)
    pauses = _find_pauses(words, min_ms=pause_min_ms) if remove_pauses else []
    pause_lines = [
        f'{p["start_ms"]} {p["end_ms"]} {p["duration_ms"]}ms  ("{p["before"]}" \u2192 "{p["after"]}")'
        for p in pauses[:100]
    ]
    pause_text = "\n".join(pause_lines) if pause_lines else "None detected"

    print(f"Pre-detected {len(fillers)} fillers, {len(pauses)} pauses")

    prompt = f"""You are an experienced podcast editor. Your task is to clean and polish this podcast episode while keeping a natural, human flow. The goal is clean, confident, and human \u2014 NOT over-edited.

UTTERANCE TRANSCRIPT (for context):
{utt_text}

PRE-DETECTED FILLER WORDS (format: start_ms end_ms "word" speaker):
{filler_text}

PRE-DETECTED PAUSES >{pause_min_ms}ms (format: start_ms end_ms duration before\u2192after):
{pause_text}

RECORDING CONTEXT:
{preset_cfg.get('claude_hint', '')}

CUSTOM INSTRUCTIONS:
{custom_instructions if custom_instructions else "None"}

EDITING PHILOSOPHY:
- Be CONSERVATIVE. Only cut things you are highly confident should be removed. A slightly imperfect but natural-sounding podcast is far better than one with broken sentences or jarring cuts.
- Preserve personality, rhythm, and emotion. The episode should sound human, not robotic.
- Avoid abrupt or robotic cuts \u2014 edits must sound conversational and smooth.
- Do NOT over-edit. When in doubt, ALWAYS leave it in.
- NEVER make a Content Cut that would break a sentence or remove words that are needed for the sentence to make grammatical sense. If a cut would leave an incomplete or nonsensical sentence, do NOT make it.

FILLER WORD RULES:
- Remove approximately {filler_pct}% of the filler words listed above \u2014 NOT all of them.
- Keep fillers that serve as clear natural transitions between distinct thoughts.
- Remove fillers that cluster together, interrupt flow, or appear mid-sentence.
- Use the EXACT start_ms and end_ms provided. CRITICAL: Never adjust these timestamps \u2014 they are word-level boundaries from the transcription engine.
- Preserve ALL acronyms and industry-specific terms exactly as spoken \u2014 they are key terminology, not mistakes.

PAUSE RULES:
- For pauses listed above, trim so that any pause longer than 2 seconds becomes about 0.8 seconds.
- Set start_ms = pause_start_ms + 800, end_ms = pause_end_ms (keeps ~0.8s of natural pause).
- Do NOT remove short, intentional pauses used for emphasis \u2014 only trim the clearly excessive ones.

CONTENT RULES:
- STUTTERS: When the EXACT same word appears twice in a row (e.g. "so so", "part part", "just as just", "still there still"), remove the duplicate. These are speech disfluencies, not emphasis. This is the safest type of content cut.
- FALSE STARTS: Only cut when a speaker clearly abandons a sentence and restarts it. You must be very confident the restart is cleaner. If in doubt, leave both.
- Do NOT make speculative content cuts. Only cut content you are 95%+ confident should be removed.
- Do NOT rewrite or paraphrase content. Do NOT change the meaning or tone of the speaker.
- All content cuts MUST start at the beginning of a word (use the word's start_ms) and end at the end of a word (use the word's end_ms). NEVER cut mid-word.
- SAFETY CHECK: Before finalizing any Content Cut, mentally read the sentence with the cut applied. If the remaining words do not form a complete, grammatical sentence, do NOT make the cut.

PRESERVE RULES (do NOT cut these):
- TRANSITIONAL PHRASES that connect topics or introduce new points, even if they seem tangential. E.g. "I think the other thing I'd flag which maybe doesn't fall under compliance but it's part of it" — these bridge ideas and must be kept.
- AGREEMENT MARKERS between speakers like "yeah absolutely", "yeah exactly", "absolutely" — these show active engagement and are part of natural conversation flow.
- SHORT RESPONSES between speakers like "yeah" or "right" that acknowledge the other speaker — these maintain conversational rhythm and show listening. Only cut if they overlap with the other speaker's words.
- SPEAKER TRANSITIONS where one person hands off to another — keep the social glue that makes dialogue sound natural.

STRUCTURAL RULES:
- If there is pre-interview chat before the official episode begins, mark it for removal.
- If there is post-interview chat after the episode has clearly concluded, mark it for removal.
- Preserve the full interview content itself.

OUTPUT RULES:
- Add at least one "Note" decision (no start_ms/end_ms) summarizing the overall edit.
- Use ONLY ms values from the data above \u2014 never invent values.
- You MUST include at least one decision.

Call the submit_edit_decisions tool with your decisions.

Example tool input for reference:
{{"decisions": [{{"type": "Remove Filler", "description": "Remove 'um'", "start_ms": 12680, "end_ms": 12900, "confidence": 95, "rationale": "Filler word \u2014 interrupts flow"}}, {{"type": "Trim Pause", "description": "Trim 3200ms pause to ~1s", "start_ms": 15000, "end_ms": 17200, "confidence": 90, "rationale": "Excessive pause"}}, {{"type": "Content Cut", "description": "Remove false start", "start_ms": 22000, "end_ms": 23500, "confidence": 85, "rationale": "Speaker restarts sentence more clearly"}}, {{"type": "Note", "description": "Light edit \u2014 preserved natural conversational flow", "confidence": 100, "rationale": "Summary"}}]}}"""

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

    max_retries = 8
    message = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_edit_decisions"},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                wait = min(10 * (2 ** attempt), 120)
                print(f"Claude API {e.status_code}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    if message is None:
        raise Exception(f"Claude API failed after {max_retries} attempts")

    edit_decisions = None
    for block in message.content:
        if block.type == "tool_use" and block.name == "submit_edit_decisions":
            edit_decisions = block.input.get("decisions", [])
            break

    if edit_decisions is None:
        raise Exception("Claude did not return tool call \u2014 unexpected response format")

    print(f"Generated {len(edit_decisions)} edit decisions")
    return {
        "edit_decisions": edit_decisions,
        "analysis_timestamp": datetime.now().isoformat()
    }


# ============================================================================
# AUDIO EDITING — cuts with crossfade
# ============================================================================

def _snap_to_zero_crossing(audio, ms, search_radius_ms=5):
    """Snap to the nearest zero-crossing within \u00b1search_radius_ms."""
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
    Remove segments from audio using pydub with crossfade, export as WAV.
    cuts_ms: list of (start_ms, end_ms) tuples to remove.
    words: optional word dicts for snapping cuts to word boundaries.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    word_intervals = [
        (w.get('start', 0), w.get('end', 0))
        for w in (words or [])
        if w.get('end', 0) > w.get('start', 0)
    ]

    def _lands_inside_word(ms):
        for ws, we in word_intervals:
            if ws < ms < we:
                return (ws, we)
        return None

    def _safe_cut_start(ms):
        hit = _lands_inside_word(ms)
        return hit[0] if hit else ms

    def _safe_cut_end(ms):
        hit = _lands_inside_word(ms)
        return hit[1] if hit else ms

    # Clamp, sort, merge overlapping cuts
    adjusted = []
    for s, e in cuts_ms:
        s, e = int(s), int(e)
        if e <= s:
            continue
        adjusted.append((max(0, s), min(total_ms, e)))
    adjusted.sort(key=lambda x: x[0])

    merged = []
    for s, e in adjusted:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Snap to word boundaries and zero-crossings
    if word_intervals:
        for cut in merged:
            cut[0] = _safe_cut_start(cut[0])
            cut[1] = _safe_cut_end(cut[1])
    for cut in merged:
        cut[0] = _snap_to_zero_crossing(audio, cut[0])
        cut[1] = _snap_to_zero_crossing(audio, cut[1])

    # Build kept segments with crossfade
    CROSSFADE_MS = 50
    segments = []
    pos = 0
    for s, e in merged:
        if s > pos:
            segments.append(audio[pos:s])
        pos = e
    if pos < total_ms:
        segments.append(audio[pos:])

    if segments:
        edited = segments[0]
        for seg in segments[1:]:
            if len(edited) > CROSSFADE_MS and len(seg) > CROSSFADE_MS:
                edited = edited.append(seg, crossfade=CROSSFADE_MS)
            else:
                edited += seg
    else:
        edited = AudioSegment.empty()

    removed_ms = total_ms - len(edited)
    print(f"Cuts: {len(merged)} cuts, removed {removed_ms}ms, {len(edited)}ms remaining")

    wav_path = audio_path.rsplit('.', 1)[0] + '_edited.wav'
    edited.export(wav_path, format='wav')
    print(f"Exported: {wav_path} ({len(edited)}ms, {os.path.getsize(wav_path) // 1024}KB)")
    return wav_path


# ============================================================================
# CLEANVOICE — studio sound, dereverb, mouth sounds, breathing
# ============================================================================

def process_with_cleanvoice(audio_path, preset_cfg=None):
    """Send audio to Cleanvoice for Studio Sound enhancement + cleanup."""
    if preset_cfg is None:
        preset_cfg = _get_preset('zoom')
    headers = {"X-API-Key": CLEANVOICE_API_KEY}
    filename = os.path.basename(audio_path)
    file_size_mb = os.path.getsize(audio_path) / 1024 / 1024

    # Get signed upload URL
    print(f"Cleanvoice: requesting signed URL for {filename} ({file_size_mb:.1f}MB)")
    sign_resp = requests.post(
        f'https://api.cleanvoice.ai/v2/upload?filename={filename}',
        headers=headers,
        timeout=30,
    )
    if sign_resp.status_code not in (200, 201):
        raise Exception(f"Cleanvoice signed URL failed: {sign_resp.status_code} \u2014 {sign_resp.text[:300]}")

    signed_url = sign_resp.json().get('signedUrl')
    if not signed_url:
        raise Exception(f"No signedUrl in Cleanvoice response: {sign_resp.text[:300]}")

    # Upload file
    print("Cleanvoice: uploading...")
    with open(audio_path, 'rb') as f:
        put_resp = requests.put(signed_url, data=f, headers={'Content-Type': 'audio/wav'}, timeout=300)
    if put_resp.status_code not in (200, 201):
        raise Exception(f"Cleanvoice upload failed: {put_resp.status_code} \u2014 {put_resp.text[:300]}")

    # Create edit job
    edit_resp = requests.post(
        'https://api.cleanvoice.ai/v2/edits',
        headers={**headers, 'Content-Type': 'application/json'},
        json={
            "input": {
                "files": [signed_url.split('?')[0]],
                "config": {
                    "remove_noise": preset_cfg.get('remove_noise', True),
                    "fillers": True,
                    "long_silences": False,
                    "studio_sound": preset_cfg.get('studio_sound', 'nightly'),
                }
            }
        },
        timeout=30,
    )
    if edit_resp.status_code not in (200, 201):
        raise Exception(f"Cleanvoice edit failed: {edit_resp.status_code} \u2014 {edit_resp.text[:300]}")

    edit_id = edit_resp.json().get('id')
    if not edit_id:
        raise Exception(f"No edit ID in Cleanvoice response: {edit_resp.text[:300]}")
    print(f"Cleanvoice: edit started: {edit_id}")

    # Poll for completion (max 10 minutes)
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(10)
        check = requests.get(
            f'https://api.cleanvoice.ai/v2/edits/{edit_id}',
            headers=headers,
            timeout=30,
        )
        if check.status_code != 200:
            raise Exception(f"Cleanvoice status failed: {check.status_code} \u2014 {check.text[:300]}")

        data = check.json()
        status = data.get('status', '')
        print(f"Cleanvoice: status = {status}")

        if status in ('completed', 'SUCCESS', 'done'):
            result = data.get('result', {})
            download_url = (
                result.get('download_url') or result.get('url') or
                data.get('download_url') or data.get('output', {}).get('url')
            )
            if not download_url:
                outputs = data.get('output', {}).get('files', [])
                if outputs:
                    download_url = outputs[0] if isinstance(outputs[0], str) else outputs[0].get('url')
            if not download_url:
                raise Exception(f"Cleanvoice: no download URL: {json.dumps(data)[:500]}")

            dl = requests.get(download_url, timeout=300)
            if dl.status_code != 200:
                raise Exception(f"Cleanvoice download failed: {dl.status_code}")

            output_path = audio_path.rsplit('.', 1)[0] + '_cleanvoice.wav'
            with open(output_path, 'wb') as out:
                out.write(dl.content)
            print(f"Cleanvoice: saved {output_path} ({len(dl.content) // 1024}KB)")
            return output_path

        if status in ('ERROR', 'error', 'failed'):
            raise Exception(f"Cleanvoice failed: {json.dumps(data)[:500]}")

    raise Exception("Cleanvoice timed out after 10 minutes")


# ============================================================================
# EDIT PIPELINE — background job
# ============================================================================

def _run_edit_job(job_id, audio_path, cuts_ms, transcript_id=None, preset_cfg=None):
    """
    Background thread: cuts \u2192 Cleanvoice Studio Sound \u2192 MP3 export.
    Cleanvoice handles dereverb, noise, enhancement, and mouth sounds.
    """
    if preset_cfg is None:
        preset_cfg = _get_preset('zoom')
    active_stage = 'init'
    try:
        # Fetch word timestamps for smart cut snapping
        words = None
        if transcript_id:
            try:
                transcript_data = get_transcription(transcript_id)
                words = transcript_data.get('words', [])
                print(f"Fetched {len(words)} word timestamps for cut snapping")
            except Exception as e:
                print(f"Warning: could not fetch word timestamps: {e}")

        # Stage 1: Apply cuts
        active_stage = 'cutting'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'cutting'
        wav_path = apply_audio_edits(audio_path, cuts_ms, words=words)

        # Stage 2: Cleanvoice Studio Sound — dereverb, enhance, mouth sounds
        if CLEANVOICE_API_KEY and preset_cfg.get('studio_sound'):
            active_stage = 'cleanvoice'
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'cleanvoice'
            wav_path = process_with_cleanvoice(wav_path, preset_cfg)

        # Export as MP3
        from pydub import AudioSegment
        audio = AudioSegment.from_file(wav_path)
        mp3_path = wav_path.rsplit('.', 1)[0] + '_final.mp3'
        audio.export(mp3_path, format='mp3', bitrate='192k')
        final_path = mp3_path

        with _edit_jobs_lock:
            _edit_jobs[job_id] = {
                'status': 'completed',
                'path': final_path,
                'is_mp3': True,
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


# ============================================================================
# REPORT
# ============================================================================

def generate_edit_report(filename, transcript_data, edit_analysis, preset_cfg):
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
PRESET
--------------------------------------------------------------------------------
Filler removal: {preset_cfg.get('filler_pct', 60)}%
Pause threshold: {preset_cfg.get('pause_min_ms', 2000)}ms
Studio sound: {preset_cfg.get('studio_sound', 'nightly')}
Noise removal: {preset_cfg.get('remove_noise', True)}

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
        ts = f"{format_timestamp(d['start_ms'])} \u2192 {format_timestamp(d['end_ms'])}" if 'start_ms' in d else "N/A"
        report += f"[{i}] {ts} \u2014 {d.get('type', 'N/A')}\n"
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
        "version": "5.0.0",
    })


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Step 1: Receive audio, upload to AssemblyAI, start transcription."""
    if request.method == 'OPTIONS':
        return '', 204

    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use MP3, WAV, M4A, AAC, or OGG"}), 400

    print(f"Upload: {file.filename}")
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'mp3'

    try:
        file_data = file.read()
        upload_url = stream_bytes_to_assemblyai(file_data)
        transcript_id = start_transcription(upload_url)

        tmp_path = f"/tmp/{transcript_id}.{ext}"
        with open(tmp_path, 'wb') as f:
            f.write(file_data)

        return jsonify({
            "success": True,
            "transcript_id": transcript_id,
            "filename": secure_filename(file.filename),
            "message": "File uploaded and transcription started",
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/transcription-status/<transcript_id>', methods=['GET'])
def transcription_status(transcript_id):
    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500
    try:
        data = get_transcription(transcript_id)
        status = data.get("status")
        return jsonify({
            "status": status,
            "error": data.get("error") if status == "error" else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _run_analysis_job(job_id, transcript_data, filename, preset_cfg, custom_instructions):
    """Background thread: run Claude analysis."""
    try:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'analyzing'

        edit_analysis = analyze_transcript_with_claude(transcript_data, preset_cfg, custom_instructions)
        report = generate_edit_report(filename, transcript_data, edit_analysis, preset_cfg)
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
                "confidence": transcript_data.get('confidence', 0) or 0,
            }
        }
        with _jobs_lock:
            _jobs[job_id] = {'status': 'completed', 'result': result}
        print(f"Analysis job {job_id} complete: {cuts_count} cuts")

    except anthropic.APIStatusError as e:
        err = "Claude API is temporarily overloaded. Please try again." if e.status_code == 529 else str(e)
        with _jobs_lock:
            _jobs[job_id] = {'status': 'error', 'error': err, 'retryable': e.status_code == 529}
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _jobs_lock:
            _jobs[job_id] = {'status': 'error', 'error': str(e)}


@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_podcast():
    """Step 2: Fetch transcript, kick off Claude analysis in background."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
            return jsonify({"error": "API keys not configured on server"}), 500

        data = request.json
        transcript_id = data.get('transcript_id')
        filename = data.get('filename', 'episode')
        preset_cfg = _get_preset(data.get('preset', 'zoom'))
        custom_instructions = data.get('customInstructions', '')

        if not transcript_id:
            return jsonify({"error": "No transcript_id provided"}), 400

        transcript_data = get_transcription(transcript_id)
        status = transcript_data.get("status")
        if status == "error":
            raise Exception(f"Transcription failed: {transcript_data.get('error', 'Unknown error')}")
        if status != "completed":
            return jsonify({"error": f"Transcription not ready (status: {status})."}), 400

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {'status': 'pending'}

        threading.Thread(
            target=_run_analysis_job,
            args=(job_id, transcript_data, filename, preset_cfg, custom_instructions),
            daemon=True,
        ).start()

        return jsonify({"success": True, "job_id": job_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/process-status/<job_id>', methods=['GET'])
def process_status(job_id):
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
    """Step 3: Start async audio editing job."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        transcript_id = data.get('transcript_id')
        cuts = data.get('cuts', [])
        preset_cfg = _get_preset(data.get('preset', 'zoom'))

        if not transcript_id:
            return jsonify({"error": "No transcript_id provided"}), 400
        if not CLEANVOICE_API_KEY and preset_cfg.get('studio_sound'):
            return jsonify({"error": "CLEANVOICE_API_KEY not configured \u2014 cannot enhance audio"}), 500

        audio_path = None
        for ext in ALLOWED_EXTENSIONS:
            path = f"/tmp/{transcript_id}.{ext}"
            if os.path.exists(path):
                audio_path = path
                break
        if not audio_path:
            return jsonify({"error": "Audio file not found. Please re-upload."}), 404

        cuts_ms = [
            (c['start_ms'], c['end_ms'])
            for c in cuts
            if 'start_ms' in c and 'end_ms' in c
        ]
        print(f"Edit job: {len(cuts_ms)} cuts, preset: {data.get('preset', 'zoom')}")

        job_id = str(uuid.uuid4())
        with _edit_jobs_lock:
            _edit_jobs[job_id] = {'status': 'pending'}

        threading.Thread(
            target=_run_edit_job,
            args=(job_id, audio_path, cuts_ms, transcript_id, preset_cfg),
            daemon=True,
        ).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/edit-audio-status/<job_id>', methods=['GET'])
def edit_audio_status(job_id):
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
    with _edit_jobs_lock:
        job = _edit_jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({"error": "File not ready"}), 404
    return send_file(job['path'], as_attachment=True, download_name='edited_podcast.mp3', mimetype='audio/mpeg')


@app.route('/api/transcript-download/<transcript_id>', methods=['GET'])
def transcript_download(transcript_id):
    """Return the transcript as a formatted text file with speaker labels."""
    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500
    try:
        data = get_transcription(transcript_id)
        if data.get('status') != 'completed':
            return jsonify({"error": "Transcript not ready"}), 404

        utterances = data.get('utterances', [])
        lines = []
        for utt in utterances:
            start = format_timestamp(utt.get('start', 0))
            speaker = utt.get('speaker', '?')
            text = utt.get('text', '')
            lines.append(f"[{start}] Speaker {speaker}: {text}")

        transcript_text = "\n\n".join(lines)
        return Response(
            transcript_text,
            mimetype='text/plain',
            headers={'Content-Disposition': 'attachment; filename=transcript.txt'}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
        "claude_configured": bool(CLAUDE_API_KEY),
        "cleanvoice_configured": bool(CLEANVOICE_API_KEY),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
