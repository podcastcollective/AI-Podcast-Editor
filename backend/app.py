"""
Flask Backend API for AI Podcast Editor
Handles file uploads, transcription, Claude analysis, and audio editing.
"""

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import subprocess
import time
import threading
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import re
import hashlib
import base64
import urllib.parse
import anthropic
import requests

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}

# API keys
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
_adobe_token_lock = threading.Lock()
_adobe_token: str | None = os.environ.get('ADOBE_ENHANCE_TOKEN')
_adobe_token_refresh_needed: bool = False


def get_adobe_token():
    with _adobe_token_lock:
        return _adobe_token


def set_adobe_token(token):
    global _adobe_token, _adobe_token_refresh_needed
    with _adobe_token_lock:
        _adobe_token = token
        _adobe_token_refresh_needed = False


def _mark_token_used():
    global _adobe_token_refresh_needed
    with _adobe_token_lock:
        _adobe_token_refresh_needed = True


if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY and CLAUDE_API_KEY are required")
if _adobe_token:
    print("Adobe Enhance Speech: configured (auto-enhance enabled)")
else:
    print("Adobe Enhance Speech: not configured (manual enhance mode)")

# In-memory job stores. Gunicorn must use threads (not processes) so these are shared.
_jobs: dict = {}
_jobs_lock = threading.Lock()
_edit_jobs: dict = {}
_edit_jobs_lock = threading.Lock()
# Multi-track metadata: preserved between upload and edit phases for per-track cutting
_multitrack_meta: dict = {}  # transcript_id → {track_paths, labels, enhance_mixes, alignment_info}
_multitrack_meta_lock = threading.Lock()


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
# PRESETS — control Claude editing behaviour
# ============================================================================

PRESETS = {
    'studio': {
        'remove_fillers': True,
        'filler_pct': 40,           # remove fewer fillers — studio audio is clean
        'remove_pauses': True,
        'pause_min_ms': 2500,       # only trim very long pauses
        'pause_target_ms': 800,
        'claude_hint': 'This is a studio recording with clean audio. Be very conservative with cuts — only remove clear disfluencies. The audio quality is already good, so preserve the natural sound.',
    },
    'zoom': {
        'remove_fillers': True,
        'filler_pct': 60,
        'remove_pauses': True,
        'pause_min_ms': 2000,
        'pause_target_ms': 800,
        'claude_hint': 'This is a remote/Zoom recording. Standard editing — remove clear filler words and trim long pauses while keeping conversational flow.',
    },
    'solo': {
        'remove_fillers': True,
        'filler_pct': 80,           # aggressive filler removal for narration
        'remove_pauses': True,
        'pause_min_ms': 600,        # catch shorter pauses for tight narration
        'pause_target_ms': 300,     # trim to ~0.3s for polished delivery
        'claude_hint': 'This is solo narration. Be more aggressive with filler word removal since there is no conversation to preserve. Tighten pauses for a polished delivery.',
    },
    'raw': {
        'remove_fillers': True,
        'filler_pct': 80,
        'remove_pauses': True,
        'pause_min_ms': 1500,
        'pause_target_ms': 800,
        'claude_hint': 'This is a rough recording that needs heavy cleanup. Be aggressive with filler removal and pause trimming. Look for false starts and repeated sentences to cut.',
    },
}


def _get_preset(name):
    """Return preset config, defaulting to 'zoom' for unknown values."""
    return PRESETS.get(name, PRESETS['zoom'])


def _analyze_audio(audio_path, transcript_data):
    """
    Analyze audio + transcript metadata to auto-detect the best preset.
    Returns (preset_name, metrics_dict).
    Uses ffmpeg to extract a short sample to avoid OOM on large files.
    """
    from pydub import AudioSegment

    # Extract first 5 min as mono 16kHz WAV (~9MB) instead of loading full file (~600MB)
    MAX_ANALYSIS_S = 300
    MAX_ANALYSIS_MS = MAX_ANALYSIS_S * 1000
    sample_path = audio_path + '_analysis_sample.wav'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path, '-t', str(MAX_ANALYSIS_S),
             '-ac', '1', '-ar', '16000', '-f', 'wav', sample_path],
            capture_output=True, check=True, timeout=60,
        )
        audio = AudioSegment.from_file(sample_path)
    finally:
        if os.path.exists(sample_path):
            os.remove(sample_path)

    # Speaker count from utterances
    utterances = transcript_data.get('utterances', [])
    speakers = set(u.get('speaker') for u in utterances if u.get('speaker'))
    speaker_count = len(speakers)

    # Noise floor: measure RMS of gaps between words (silence segments)
    words = transcript_data.get('words', [])
    gap_dbfs_values = []
    for i in range(len(words) - 1):
        gap_start = words[i].get('end', 0)
        gap_end = words[i + 1].get('start', 0)
        gap_ms = gap_end - gap_start
        if gap_ms >= 200 and gap_end <= MAX_ANALYSIS_MS:
            segment = audio[gap_start:gap_end]
            if len(segment) > 0:
                if segment.dBFS != float('-inf'):
                    gap_dbfs_values.append(segment.dBFS)
            if len(gap_dbfs_values) >= 30:  # sample up to 30 gaps
                break
    noise_floor = sum(gap_dbfs_values) / len(gap_dbfs_values) if gap_dbfs_values else -35.0

    # Transcription confidence
    confidence = transcript_data.get('confidence', 0) or 0

    # Dynamic range
    dynamic_range = audio.max_dBFS - audio.dBFS if audio.dBFS != float('-inf') else 0

    # Decision tree
    if speaker_count <= 1:
        preset_name = 'solo'
    elif noise_floor < -42 and confidence >= 0.90 and dynamic_range > 10:
        preset_name = 'studio'
    elif noise_floor > -28 or confidence < 0.78:
        preset_name = 'raw'
    else:
        preset_name = 'zoom'

    import math
    _safe = lambda v, default=0: default if (math.isinf(v) or math.isnan(v)) else v
    metrics = {
        'speakers': speaker_count,
        'noise_floor': round(_safe(noise_floor, -35.0), 1),
        'confidence': round(_safe(confidence), 3),
        'dynamic_range': round(_safe(dynamic_range), 1),
    }
    print(f"Auto-detect: speakers={speaker_count}, noise_floor={noise_floor:.1f}dB, "
          f"confidence={confidence:.3f}, dynamic_range={dynamic_range:.1f}dB \u2192 {preset_name}")
    return preset_name, metrics


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
            speaker_before = words[i].get('speaker', '?')
            speaker_after = words[i + 1].get('speaker', '?')
            pauses.append({
                'start_ms': gap_start,
                'end_ms': gap_end,
                'duration_ms': gap_ms,
                'before': words[i].get('text', ''),
                'after': words[i + 1].get('text', ''),
                'speaker_before': speaker_before,
                'speaker_after': speaker_after,
                'speaker_change': speaker_before != speaker_after,
            })
    return pauses


def format_timestamp(milliseconds):
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def analyze_transcript_with_claude(transcript_data, preset_cfg, custom_instructions="", is_multitrack=False):
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
    pause_target_ms = preset_cfg.get('pause_target_ms', 800)
    pauses = _find_pauses(words, min_ms=pause_min_ms) if remove_pauses else []
    pause_lines = []
    for p in pauses[:100]:
        line = f'{p["start_ms"]} {p["end_ms"]} {p["duration_ms"]}ms  ("{p["before"]}" \u2192 "{p["after"]}")'
        if is_multitrack and p.get('speaker_change'):
            line += f'  [SPEAKER CHANGE: {p["speaker_before"]}\u2192{p["speaker_after"]}]'
        pause_lines.append(line)
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
- For pauses listed above, trim so that any pause longer than {pause_min_ms}ms becomes about {pause_target_ms/1000:.1f} seconds.
- Set start_ms = pause_start_ms + {pause_target_ms}, end_ms = pause_end_ms (keeps ~{pause_target_ms/1000:.1f}s of natural pause).
- Trim ALL pauses in the list above \u2014 they have already been filtered by threshold, so every one should be trimmed.
- Do NOT remove short, intentional pauses used for emphasis \u2014 only trim the clearly excessive ones.
{"" if not is_multitrack else """
MULTI-TRACK RULES (CRITICAL — this audio was mixed from separate speaker tracks):
- SPEAKER TRANSITION PAUSES: When a pause occurs between two DIFFERENT speakers, do NOT trim it shorter than 1.5 seconds. Trimming speaker transitions too aggressively causes speakers to sound like they are talking over each other because each track contains background audio from the recording environment.
- FILLER WORDS NEAR TRANSITIONS: Do NOT remove filler words that occur within 500ms of another speaker starting or stopping. These fillers overlap with the other speaker's audio on their track.
- Be MORE CONSERVATIVE overall with pause trimming — it is far better to have a slightly longer pause than to create speaker overlap artifacts.
- If a pause is between the SAME speaker's sentences, normal trimming rules apply.
"""}
CONTENT RULES:
- STUTTERS: When the EXACT same word appears twice in a row (e.g. "so so", "part part", "just as just", "still there still"), remove the duplicate. These are speech disfluencies, not emphasis. This is the safest type of content cut.
- FALSE STARTS: Only cut when a speaker clearly abandons a sentence and restarts it. You must be very confident the restart is cleaner. If in doubt, leave both.
- Do NOT make speculative content cuts. Only cut content you are 95%+ confident should be removed.
- Do NOT rewrite or paraphrase content. Do NOT change the meaning or tone of the speaker.
- All content cuts MUST start at the beginning of a word (use the word's start_ms) and end at the end of a word (use the word's end_ms). NEVER cut mid-word.
- SAFETY CHECK: Before finalizing any Content Cut, mentally read the sentence with the cut applied. If the remaining words do not form a complete, grammatical sentence, do NOT make the cut.

PRESERVE RULES (do NOT cut these):
- TRANSITIONAL PHRASES that connect topics or introduce new points, even if they seem tangential. E.g. "I think the other thing I'd flag which maybe doesn't fall under compliance but it's part of it" \u2014 these bridge ideas and must be kept.
- AGREEMENT MARKERS between speakers like "yeah absolutely", "yeah exactly", "absolutely" \u2014 these show active engagement and are part of natural conversation flow.
- SHORT RESPONSES between speakers like "yeah" or "right" that acknowledge the other speaker \u2014 these maintain conversational rhythm and show listening. Only cut if they overlap with the other speaker's words.
- SPEAKER TRANSITIONS where one person hands off to another \u2014 keep the social glue that makes dialogue sound natural.

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

    models = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    max_retries = 3
    message = None
    for model in models:
        for attempt in range(max_retries):
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=16000,
                    tools=tools,
                    tool_choice={"type": "tool", "name": "submit_edit_decisions"},
                    messages=[{"role": "user", "content": prompt}],
                )
                print(f"Claude analysis succeeded with {model}")
                break
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 529) and attempt < max_retries - 1:
                    wait = min(10 * (2 ** attempt), 30)
                    print(f"{model} API {e.status_code}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                elif e.status_code in (429, 529):
                    print(f"{model} failed after {max_retries} attempts, trying next model")
                    break
                else:
                    raise
        if message is not None:
            break
    if message is None:
        raise Exception("Claude API failed \u2014 all models overloaded")

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
# AUDIO EDITING — ffmpeg-based cuts (no pydub, avoids OOM on large files)
# ============================================================================

def apply_audio_edits(audio_path, cuts_ms, words=None):
    """
    Remove segments from audio using ffmpeg filter_complex, export as WAV.
    Streams the file through ffmpeg instead of loading into memory.
    cuts_ms: list of (start_ms, end_ms) tuples to remove.
    words: optional word dicts for snapping cuts to word boundaries.
    """
    # Get audio duration with ffprobe
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', audio_path],
        capture_output=True, text=True, check=True, timeout=30,
    )
    total_ms = int(float(probe.stdout.strip()) * 1000)

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
        """If cut start lands inside a word, push it to AFTER the word (preserve it)."""
        hit = _lands_inside_word(ms)
        return hit[1] if hit else ms

    def _safe_cut_end(ms):
        """If cut end lands inside a word, pull it to BEFORE the word (preserve it)."""
        hit = _lands_inside_word(ms)
        return hit[0] if hit else ms

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

    # Snap to word boundaries (zero-crossing snap removed — 5ms fades prevent clicks)
    if word_intervals:
        for cut in merged:
            cut[0] = _safe_cut_start(cut[0])
            cut[1] = _safe_cut_end(cut[1])

    # Calculate keep segments
    keeps = []
    pos = 0
    for s, e in merged:
        if s > pos:
            keeps.append((pos, s))
        pos = e
    if pos < total_ms:
        keeps.append((pos, total_ms))

    removed_ms = sum(e - s for s, e in merged)
    remaining_ms = total_ms - removed_ms
    print(f"Cuts: {len(merged)} cuts, removed {removed_ms}ms, {remaining_ms}ms remaining")

    if not keeps:
        keeps = [(0, min(1000, total_ms))]

    # Build ffmpeg filter_complex: trim each keep segment, apply micro-fades, concat
    FADE_S = 0.010  # 10ms micro-fade at cut boundaries (5ms caused audible clicks)
    filter_parts = []
    for i, (start, end) in enumerate(keeps):
        start_s = start / 1000
        end_s = end / 1000
        duration_s = end_s - start_s

        # atrim extracts segment, asetpts resets timestamps to 0
        chain = [f"atrim={start_s}:{end_s}", "asetpts=N/SR/TB"]

        # Micro-fades at cut boundaries only (not episode start/end)
        if i > 0 and duration_s > FADE_S:
            chain.append(f"afade=t=in:d={FADE_S}")
        if i < len(keeps) - 1 and duration_s > FADE_S:
            chain.append(f"afade=t=out:st={duration_s - FADE_S}:d={FADE_S}")

        filter_parts.append(f"[0:a]{','.join(chain)}[s{i}]")

    concat_inputs = "".join(f"[s{i}]" for i in range(len(keeps)))
    full_filter = ";".join(filter_parts)
    full_filter += f";{concat_inputs}concat=n={len(keeps)}:v=0:a=1[out]"

    # Write filter to script file to avoid command-line length limits
    filter_path = audio_path + '_filter.txt'
    wav_path = audio_path.rsplit('.', 1)[0] + '_edited.wav'
    try:
        with open(filter_path, 'w') as f:
            f.write(full_filter)

        result = subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path,
             '-filter_complex_script', filter_path,
             '-map', '[out]', '-f', 'wav', wav_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise Exception(f"ffmpeg edit failed: {result.stderr[-500:]}")
    finally:
        if os.path.exists(filter_path):
            os.remove(filter_path)

    print(f"Exported: {wav_path} ({remaining_ms}ms, {os.path.getsize(wav_path) // 1024}KB)")
    return wav_path


# ============================================================================
# MULTI-TRACK — align and combine separate speaker tracks
# ============================================================================

def _measure_noise_floor(audio_path, duration_s=2):
    """Measure RMS level of the first N seconds to estimate noise floor."""
    result = subprocess.run(
        ['ffmpeg', '-i', audio_path, '-t', str(duration_s),
         '-af', 'volumedetect', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=30,
    )
    for line in result.stderr.split('\n'):
        m = re.search(r'mean_volume:\s*([-\d.]+)', line)
        if m:
            return float(m.group(1))
    return -35.0  # fallback


def _detect_speech_onset(audio_path):
    """
    Detect where speech begins in an audio track using ffmpeg silencedetect.
    Uses adaptive threshold: measures noise floor of first 2s, then sets
    silence threshold 5dB above it so crosstalk/room tone doesn't mask silence.
    Returns onset in milliseconds. Returns 0 if speech starts immediately.
    """
    # Adaptive threshold: noise floor + 5dB, clamped to [-40, -20] range
    noise_floor = _measure_noise_floor(audio_path)
    threshold = max(-40, min(-20, noise_floor + 5))
    print(f"Speech onset detection for {os.path.basename(audio_path)}: "
          f"noise_floor={noise_floor:.1f}dB, threshold={threshold:.1f}dB")

    result = subprocess.run(
        ['ffmpeg', '-i', audio_path, '-af',
         f'silencedetect=noise={threshold}dB:d=0.5', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=120,
    )
    # Parse silence events to find speech onset.
    # silence_start near 0 + silence_end = audio starts with silence, onset = silence_end.
    # silence_start well above 0 = audio starts with speech, onset = 0.
    first_start = None
    first_end = None
    for line in result.stderr.split('\n'):
        if first_start is None:
            m = re.search(r'silence_start:\s*([\d.]+)', line)
            if m:
                first_start = float(m.group(1))
        if first_end is None:
            m = re.search(r'silence_end:\s*([\d.]+)', line)
            if m:
                first_end = float(m.group(1))
        if first_start is not None and first_end is not None:
            break

    fname = os.path.basename(audio_path)
    if first_start is not None and first_start < 0.1 and first_end is not None:
        # Audio starts with silence; speech begins at silence_end
        print(f"Speech onset in {fname}: {first_end:.3f}s (after {first_end:.1f}s leading silence)")
        return int(first_end * 1000)
    if first_start is not None and first_start >= 0.1:
        # First silence is well into the track — speech starts at 0
        print(f"Speech onset in {fname}: 0ms (starts with speech)")
        return 0
    # No silence detected at all
    print(f"Speech onset in {fname}: 0ms (no silence detected)")
    return 0


def _get_track_duration(audio_path):
    """Get track duration in seconds via ffprobe."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', audio_path],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(probe.stdout.strip())


def _measure_loudness(audio_path):
    """Measure integrated loudness (LUFS) and true peak (dBTP) using ffmpeg loudnorm first-pass."""
    result = subprocess.run(
        ['ffmpeg', '-hide_banner', '-i', audio_path, '-af',
         'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=120,
    )
    import json as _json
    stderr = result.stderr
    json_start = stderr.rfind('{')
    json_end = stderr.rfind('}') + 1
    if json_start >= 0 and json_end > json_start:
        data = _json.loads(stderr[json_start:json_end])
        lufs = float(data.get('input_i', '-24'))
        tp = float(data.get('input_tp', '-1'))
        print(f"Loudness of {os.path.basename(audio_path)}: {lufs:.1f} LUFS, peak={tp:.1f} dBTP")
        return lufs, tp
    print(f"Loudness measurement failed for {os.path.basename(audio_path)}, using defaults")
    return -24.0, -1.0


def _combine_tracks(track_paths, labels=None):
    """
    Normalize levels, optionally align, and mix multiple audio tracks into WAV.
    Only aligns when track durations differ by >2s (different recording start
    times). Same-duration tracks are assumed to be from the same session and
    are mixed as-is.
    Returns (combined_path, alignment_info).
    """
    n = len(track_paths)
    if n < 2:
        raise ValueError("Need at least 2 tracks to combine")

    # Check if tracks need alignment (different start times vs same session)
    durations = [_get_track_duration(p) for p in track_paths]
    duration_spread = max(durations) - min(durations)
    needs_alignment = duration_spread > 2.0
    print(f"Track durations: {[f'{d:.1f}s' for d in durations]}, "
          f"spread={duration_spread:.1f}s, alignment={'yes' if needs_alignment else 'no (same session)'}")

    onsets = []
    delays = []
    if needs_alignment:
        onsets = [_detect_speech_onset(p) for p in track_paths]
        max_onset = max(onsets)
        delays = [max_onset - onset for onset in onsets]
    else:
        onsets = [0] * n
        delays = [0] * n

    alignment_info = []
    for i in range(n):
        label = (labels[i] if labels and i < len(labels) else None) or f"Track {i + 1}"
        alignment_info.append({
            'track': i,
            'label': label,
            'onset_ms': onsets[i],
            'delay_ms': delays[i],
            'aligned': needs_alignment,
        })
        if needs_alignment:
            print(f"Track {i} ({label}): onset={onsets[i]}ms, delay={delays[i]}ms")

    # Measure loudness of each track (first-pass), then apply simple gain.
    # Using volume= instead of loudnorm avoids: (a) loudnorm upsampling to 192kHz,
    # (b) variable lookahead latency causing inter-track timing drift.
    target_lufs = -16
    peak_ceiling = -3  # dBTP ceiling per track — leaves headroom for summing
    measurements = [_measure_loudness(p) for p in track_paths]
    gains = []
    for lufs, tp in measurements:
        lufs_gain = target_lufs - lufs
        # Cap gain so peaks don't exceed ceiling (prevents clipping when summed)
        max_gain = peak_ceiling - tp
        gain = min(lufs_gain, max_gain)
        gains.append(gain)

    # Build ffmpeg filter_complex
    # Per-track: normalize format → volume gain → optional delay → pad to equal length
    # Padding prevents ffmpeg amix assertion crash when streams end at different times.
    max_dur_s = max(durations[i] + delays[i] / 1000 for i in range(n))
    filter_parts = []
    for i in range(n):
        chain = f'[{i}:a]aformat=sample_rates=48000:channel_layouts=mono'
        gain_db = gains[i]
        if abs(gain_db) > 0.5:
            chain += f',volume={gain_db:.1f}dB'
            print(f"Track {i}: applying {gain_db:+.1f}dB gain ({measurements[i][0]:.1f} LUFS, peak {measurements[i][1]:.1f} dBTP)")
        if delays[i] > 0:
            d = delays[i]
            chain += f',adelay={d}|{d}'
        chain += f',apad=whole_dur={max_dur_s:.3f}'
        chain += f'[t{i}]'
        filter_parts.append(chain)

    mix_inputs = ''.join(f'[t{i}]' for i in range(n))
    # amix sums signals; -3dB headroom + limiter at -1dB to leave room for MP3 encoding
    filter_parts.append(
        f'{mix_inputs}amix=inputs={n}:duration=longest:dropout_transition=0:normalize=0,'
        f'volume=-3dB,alimiter=limit=0.89:attack=0.1:release=50[out]'
    )
    filter_str = ';'.join(filter_parts)

    combine_id = str(uuid.uuid4())[:8]
    combined_path = f'/tmp/combined_{combine_id}.wav'

    inputs = []
    for p in track_paths:
        inputs.extend(['-i', p])

    result = subprocess.run(
        ['ffmpeg', '-y'] + inputs + [
            '-filter_complex', filter_str,
            '-map', '[out]', '-f', 'wav', combined_path,
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise Exception(f"ffmpeg combine failed: {result.stderr[-500:]}")

    size_kb = os.path.getsize(combined_path) // 1024
    print(f"Combined {n} tracks: {combined_path} ({size_kb}KB)")
    return combined_path, alignment_info


# ============================================================================
# CROSSTALK SUPPRESSION — silence other speakers before Adobe enhancement
# ============================================================================

def _measure_rms_at(track_path, start_s, duration_s=0.5):
    """Measure mean RMS volume at a specific point in a track."""
    result = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', track_path,
         '-ss', str(start_s), '-t', str(duration_s),
         '-af', 'volumedetect', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=30,
    )
    for line in result.stderr.split('\n'):
        m = re.search(r'mean_volume:\s*([-\d.]+)', line)
        if m:
            return float(m.group(1))
    return -99.0


def _map_speakers_to_tracks(track_paths, words, alignment_info):
    """
    Auto-detect which AssemblyAI speaker corresponds to which track
    by measuring which track is loudest during each speaker's words.
    Returns dict: {speaker_label: track_index} or None if mapping fails.
    """
    speakers = sorted(set(w.get('speaker', '?') for w in words if w.get('speaker')))
    n_tracks = len(track_paths)
    if len(speakers) < 2 or len(speakers) > n_tracks:
        print(f"Speaker mapping: {len(speakers)} speakers, {n_tracks} tracks — skipping")
        return None

    # Sample up to 5 substantial words per speaker for volume comparison
    speaker_words = {s: [] for s in speakers}
    for w in words:
        s = w.get('speaker')
        if s and len(speaker_words.get(s, [])) < 5:
            dur_ms = w.get('end', 0) - w.get('start', 0)
            if dur_ms > 200:
                speaker_words[s].append(w)

    mapping = {}
    for speaker in speakers:
        samples = speaker_words[speaker]
        if not samples:
            continue
        track_vols = []
        for ti, track_path in enumerate(track_paths):
            delay_ms = alignment_info[ti].get('delay_ms', 0)
            volumes = []
            for w in samples:
                start_s = max(0, (w.get('start', 0) - delay_ms) / 1000.0)
                dur_s = max(0.2, (w.get('end', 0) - w.get('start', 0)) / 1000.0)
                vol = _measure_rms_at(track_path, start_s, dur_s)
                volumes.append(vol)
            avg = sum(volumes) / len(volumes) if volumes else -99
            track_vols.append(avg)
        loudest = track_vols.index(max(track_vols))
        mapping[speaker] = loudest
        print(f"Speaker {speaker} → Track {loudest} (avg dB: {[f'{v:.1f}' for v in track_vols]})")

    if len(set(mapping.values())) != len(mapping):
        print("Speaker-track mapping not 1:1, skipping crosstalk suppression")
        return None
    return mapping


def _apply_crosstalk_gate(track_path, words, own_speaker, delay_ms, pad_ms=150):
    """
    Silence regions on a track where other speakers are talking.
    Uses transcription word timestamps to identify own-speaker segments,
    then mutes everything else (crosstalk from headphones/room).

    pad_ms: padding around each word to catch natural speech transitions.
    """
    keep_ranges = []
    for w in words:
        if w.get('speaker') == own_speaker:
            ws = max(0, (w.get('start', 0) - delay_ms - pad_ms)) / 1000.0
            we = max(0, (w.get('end', 0) - delay_ms + pad_ms)) / 1000.0
            if we > ws:
                keep_ranges.append((ws, we))
    if not keep_ranges:
        print(f"No keep ranges for speaker {own_speaker}, skipping gate")
        return track_path

    # Merge overlapping ranges
    keep_ranges.sort()
    merged = []
    for s, e in keep_ranges:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Build ffmpeg filter: mute everything EXCEPT keep ranges
    between_parts = [f'between(t,{s:.3f},{e:.3f})' for s, e in merged]
    keep_expr = '+'.join(between_parts)
    filter_str = f"volume=0:enable='not({keep_expr})'"

    gated_path = track_path.rsplit('.', 1)[0] + '_gated.wav'
    filter_path = track_path + '_gate_filter.txt'
    try:
        with open(filter_path, 'w') as f:
            f.write(filter_str)
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', track_path,
             '-filter_script:a', filter_path,
             '-f', 'wav', gated_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"Crosstalk gate failed: {result.stderr[-300:]}")
            return track_path
    finally:
        if os.path.exists(filter_path):
            os.remove(filter_path)

    size_kb = os.path.getsize(gated_path) // 1024
    print(f"Crosstalk gate: {len(merged)} keep ranges for speaker {own_speaker}, "
          f"{os.path.basename(gated_path)} ({size_kb}KB)")
    return gated_path


def _run_multitrack_job(job_id, track_paths, labels, enhance_mixes):
    """Background thread: combine raw tracks, upload to AssemblyAI, preserve tracks for per-track cutting later."""
    try:
        n = len(track_paths)
        print(f"Multitrack job {job_id}: {n} tracks")

        # Combine raw tracks for transcription
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'combining'
            _edit_jobs[job_id]['progress'] = 'Mixing tracks together'
        combined_path, alignment_info = _combine_tracks(track_paths, labels)

        # Upload to AssemblyAI
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'uploading'
            _edit_jobs[job_id]['progress'] = 'Uploading to AssemblyAI'
        with open(combined_path, 'rb') as f:
            combined_data = f.read()
        upload_url = stream_bytes_to_assemblyai(combined_data)
        transcript_id = start_transcription(upload_url)

        # Save combined file with transcript_id
        final_path = f'/tmp/{transcript_id}.wav'
        os.rename(combined_path, final_path)

        # Preserve raw track files and metadata for per-track cutting in edit phase
        with _multitrack_meta_lock:
            _multitrack_meta[transcript_id] = {
                'track_paths': list(track_paths),
                'labels': list(labels),
                'enhance_mixes': list(enhance_mixes),
                'alignment_info': alignment_info,
            }
        print(f"Stored multitrack meta for {transcript_id}: {n} tracks preserved")

        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'transcribing'
            _edit_jobs[job_id]['transcript_id'] = transcript_id
            _edit_jobs[job_id]['alignment'] = alignment_info
            _edit_jobs[job_id]['progress'] = 'Waiting for transcription'

        print(f"Multitrack job {job_id} ready: transcript_id={transcript_id}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        for p in track_paths:
            if os.path.exists(p):
                os.remove(p)
        with _edit_jobs_lock:
            _edit_jobs[job_id] = {'status': 'error', 'error': str(e)}


def _cleanup_multitrack_files(transcript_id):
    """Clean up raw track files and multitrack metadata after editing."""
    with _multitrack_meta_lock:
        meta = _multitrack_meta.pop(transcript_id, None)
    if meta:
        for p in meta['track_paths']:
            if os.path.exists(p):
                os.remove(p)
                print(f"Cleaned up: {p}")


def _run_multitrack_edit_job(job_id, audio_path, cuts_ms, transcript_id,
                              intro_path=None, outro_path=None):
    """
    Background thread: Option C multi-track edit pipeline.
    Gate crosstalk → cut each track → enhance each → combine → merge → finalize.
    Requires ADOBE_ENHANCE_TOKEN — fails if not available.
    """
    current_step = 'cutting'
    try:
        # Load multitrack metadata
        with _multitrack_meta_lock:
            meta = _multitrack_meta.get(transcript_id)
        if not meta:
            raise Exception(f"No multitrack metadata found for {transcript_id}")

        track_paths = meta['track_paths']
        labels = meta['labels']
        enhance_mixes = meta['enhance_mixes']
        alignment_info = meta['alignment_info']
        n = len(track_paths)
        print(f"Multitrack edit {job_id}: {n} tracks, {len(cuts_ms)} cuts")

        # Fetch word timestamps for crosstalk gating and cut snapping
        words = None
        if transcript_id:
            try:
                transcript_data = get_transcription(transcript_id)
                words = transcript_data.get('words', [])
                print(f"Fetched {len(words)} word timestamps")
            except Exception as e:
                print(f"Warning: could not fetch word timestamps: {e}")

        # Step 0: Map speakers to tracks and suppress crosstalk
        speaker_map = None
        if words:
            speaker_map = _map_speakers_to_tracks(track_paths, words, alignment_info)
            if speaker_map:
                print(f"Speaker mapping: {speaker_map}")

        # Step 1: Gate crosstalk + cut each raw track
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'cutting'
        cut_track_paths = []
        for i, track_path in enumerate(track_paths):
            delay_ms = alignment_info[i].get('delay_ms', 0)

            # Suppress crosstalk before cutting (and before Adobe enhancement)
            gated_path = track_path
            if speaker_map and words:
                own_speaker = next((s for s, ti in speaker_map.items() if ti == i), None)
                if own_speaker:
                    print(f"Gating Track {i} ({labels[i]}): keeping speaker {own_speaker}")
                    gated_path = _apply_crosstalk_gate(track_path, words, own_speaker, delay_ms)

            # Adjust cuts from combined timeline to track-local timeline
            adjusted_cuts = [(max(0, s - delay_ms), max(0, e - delay_ms)) for s, e in cuts_ms]
            adjusted_cuts = [(s, e) for s, e in adjusted_cuts if e > s]
            print(f"Track {i} ({labels[i]}): delay={delay_ms}ms, {len(adjusted_cuts)} adjusted cuts")
            cut_path = apply_audio_edits(gated_path, adjusted_cuts, words=words)
            cut_track_paths.append(cut_path)

        # Step 2: Enhance each cut track, then normalize duration to prevent drift
        current_step = 'enhancing_tracks'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'enhancing_tracks'
        enhanced_cut_paths = []
        for i, cut_path in enumerate(cut_track_paths):
            with _edit_jobs_lock:
                _edit_jobs[job_id]['progress'] = f'Enhancing track {i + 1} of {n}'
            mix = enhance_mixes[i] if i < len(enhance_mixes) else None
            orig_dur = _get_track_duration(cut_path)
            print(f"Enhancing cut track {i + 1}/{n}: {os.path.basename(cut_path)}, mix={mix}, dur={orig_dur:.3f}s")
            enhanced_path = process_with_adobe_enhance(cut_path, enhance_mix=mix)
            # Adobe enhancement changes track duration slightly differently per track.
            # This causes inter-track drift when combined. Fix: use atempo to restore
            # each track to its pre-enhancement duration. The adjustment is typically
            # <1%, so the tempo change is imperceptible.
            enh_dur = _get_track_duration(enhanced_path)
            drift_ms = abs(enh_dur - orig_dur) * 1000
            if drift_ms > 50:  # >50ms difference — worth correcting
                ratio = enh_dur / orig_dur
                normalized_path = enhanced_path.rsplit('.', 1)[0] + '_normalized.wav'
                norm_result = subprocess.run(
                    ['ffmpeg', '-y', '-i', enhanced_path, '-af', f'atempo={ratio}',
                     '-f', 'wav', normalized_path],
                    capture_output=True, text=True, timeout=600,
                )
                if norm_result.returncode == 0:
                    print(f"Track {i}: duration {orig_dur:.3f}s → {enh_dur:.3f}s (drift={drift_ms:.0f}ms), normalized with atempo={ratio:.6f}")
                    enhanced_path = normalized_path
                else:
                    print(f"Track {i}: atempo normalization failed, using raw enhanced (drift={drift_ms:.0f}ms)")
            else:
                print(f"Track {i}: duration {orig_dur:.3f}s → {enh_dur:.3f}s (drift={drift_ms:.0f}ms, within tolerance)")
            enhanced_cut_paths.append(enhanced_path)

        # Step 3: Combine enhanced+cut tracks
        current_step = 'combining'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'combining'
            _edit_jobs[job_id]['progress'] = 'Mixing enhanced tracks'
        combined_path, _ = _combine_tracks(enhanced_cut_paths, labels)

        # Step 4: Merge intro/outro if provided
        if intro_path or outro_path:
            current_step = 'merging'
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'merging'
            combined_path = _concat_with_crossfade(combined_path, intro_path=intro_path, outro_path=outro_path)

        # Step 5: Finalize
        current_step = 'finalizing'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'finalizing'
        _finalize_audio(job_id, combined_path)

        # Cleanup
        _cleanup_multitrack_files(transcript_id)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _cleanup_multitrack_files(transcript_id)
        with _edit_jobs_lock:
            _edit_jobs[job_id] = {
                'status': 'error',
                'error': str(e),
                'failed_step': current_step,
            }


# ============================================================================
# INTRO/OUTRO MERGE — ffmpeg crossfade concatenation
# ============================================================================

def _concat_with_crossfade(episode_path, intro_path=None, outro_path=None, crossfade_s=2.0):
    """
    Concatenate intro + episode + outro with crossfades using ffmpeg acrossfade.
    Returns path to merged WAV. Skips segments that are None.
    """
    segments = []
    if intro_path:
        segments.append(intro_path)
    segments.append(episode_path)
    if outro_path:
        segments.append(outro_path)

    if len(segments) == 1:
        return episode_path  # nothing to merge

    print(f"Merging {len(segments)} segments with {crossfade_s}s crossfade")

    # Build ffmpeg filter chain: sequential acrossfade between adjacent segments
    inputs = []
    for seg in segments:
        inputs.extend(['-i', seg])

    if len(segments) == 2:
        # Simple case: one crossfade between two inputs
        filter_str = f'[0:a][1:a]acrossfade=d={crossfade_s}:c1=tri:c2=tri[out]'
    else:
        # Three segments: intro→episode crossfade, then result→outro crossfade
        filter_str = (
            f'[0:a][1:a]acrossfade=d={crossfade_s}:c1=tri:c2=tri[mid];'
            f'[mid][2:a]acrossfade=d={crossfade_s}:c1=tri:c2=tri[out]'
        )

    merged_path = episode_path.rsplit('.', 1)[0] + '_merged.wav'
    result = subprocess.run(
        ['ffmpeg', '-y'] + inputs + [
            '-filter_complex', filter_str,
            '-map', '[out]', '-f', 'wav', merged_path,
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise Exception(f"ffmpeg merge failed: {result.stderr[-500:]}")

    size_kb = os.path.getsize(merged_path) // 1024
    print(f"Merged: {merged_path} ({size_kb}KB)")
    return merged_path


# ============================================================================
# ADOBE ENHANCE SPEECH — reverse-engineered API
# ============================================================================

ADOBE_API_BASE = 'https://phonos-server-flex.adobe.io'


def _adobe_headers():
    """Base headers for authenticated Adobe API requests."""
    token = get_adobe_token()
    return {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'authorization': f'Bearer {token}' if not token.startswith('Bearer ') else token,
        'origin': 'https://podcast.adobe.com',
        'referer': 'https://podcast.adobe.com/',
        'sec-ch-ua': '"Chromium";v="137", "Not/A)Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
        'x-api-key': 'phonos-server-prod',
    }


def process_with_adobe_enhance(audio_path, enhance_mix=None):
    """Upload audio to Adobe Enhance Speech API and download enhanced result."""
    headers = _adobe_headers()

    # Read file and compute MD5 checksum
    with open(audio_path, 'rb') as f:
        file_data = f.read()
    file_size = len(file_data)
    checksum = base64.b64encode(hashlib.md5(file_data).digest()).decode('utf-8')
    filename = os.path.basename(audio_path)

    # Step 1: Get signed upload URL
    print(f"Adobe Enhance: requesting upload URL for {filename} ({file_size // 1024}KB)")
    resp = requests.post(
        f'{ADOBE_API_BASE}/rails/active_storage/direct_uploads',
        headers={**headers, 'content-type': 'application/json'},
        json={
            'blob': {
                'filename': filename,
                'content_type': 'audio/wav',
                'byte_size': file_size,
                'checksum': checksum,
            }
        },
    )
    if resp.status_code != 200:
        raise Exception(f'Adobe upload URL request failed: {resp.status_code} - {resp.text[:200]}')
    upload_data = resp.json()
    signed_id = upload_data['signed_id']
    signed_url = upload_data['direct_upload']['url']
    print(f"Adobe Enhance: got signed upload URL, signed_id={signed_id[:20]}...")

    # Step 2: Upload file to signed URL (no auth needed — pre-signed)
    upload_headers = {
        'Content-Length': str(file_size),
        'Content-Md5': checksum,
        'Content-Type': 'audio/wav',
        'Content-Disposition': f'inline; filename="{urllib.parse.quote(filename)}"; filename*=UTF-8\'\'{urllib.parse.quote(filename)}',
        'Accept': '*/*',
        'Origin': 'https://podcast.adobe.com',
        'Referer': 'https://podcast.adobe.com/',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
    }
    resp = requests.put(signed_url, headers=upload_headers, data=file_data)
    if resp.status_code not in (200, 204):
        raise Exception(f'Adobe file upload failed: {resp.status_code} - {resp.text[:200]}')
    print("Adobe Enhance: file uploaded successfully")

    # Step 3: Create enhancement job (v2 model with stem separation)
    track_id = str(uuid.uuid4())
    timestamp_ms = str(int(time.time() * 1000))
    resp = requests.post(
        f'{ADOBE_API_BASE}/api/v1/enhance_speech_tracks',
        headers={**headers, 'content-type': 'application/json'},
        params={'time': timestamp_ms},
        json={
            'id': track_id,
            'track_name': filename,
            'model_version': 'v2',
            'signed_id': signed_id,
        },
    )
    if resp.status_code not in (200, 201):
        raise Exception(f'Adobe enhance job creation failed: {resp.status_code} - {resp.text[:200]}')
    print(f"Adobe Enhance: v2 enhancement job created, track_id={track_id}")

    # Step 4: Poll merged_media for completion (max 180 attempts x 5s = 15 min)
    for attempt in range(180):
        time.sleep(5)
        timestamp_ms = str(int(time.time() * 1000))
        resp = requests.get(
            f'{ADOBE_API_BASE}/api/v1/enhance_speech_tracks/{track_id}/merged_media',
            headers=headers,
            params={'time': timestamp_ms},
        )
        if resp.status_code == 200:
            data = resp.json()
            if data and 'url' in data:
                break
        elif resp.status_code == 204:
            if attempt % 6 == 0:
                print(f"Adobe Enhance: still processing... ({attempt * 5}s elapsed)")
            continue
        else:
            raise Exception(f'Adobe enhance poll failed: {resp.status_code} - {resp.text[:200]}')
    else:
        raise Exception('Adobe Enhance Speech timed out after 15 minutes')
    print(f"Adobe Enhance: v2 processing complete")

    # Step 5: Create export with stem mix (configurable via enhance_mix)
    # enhanced_speech + isolated_speech = 1.0 (enhanced vs original speech balance)
    # background and music are gain multipliers (0.0 - 1.0)
    mix = enhance_mix or {}
    speech_pct = mix.get('speech', 90)
    bg_pct = mix.get('background', 10)
    music_pct = mix.get('music', 10)
    enhanced_speech_gain = speech_pct / 100.0
    isolated_speech_gain = 1.0 - enhanced_speech_gain
    background_gain = bg_pct / 100.0
    music_gain = music_pct / 100.0
    mix_label = f"{speech_pct}/{bg_pct}/{music_pct}"
    print(f"Adobe Enhance: using {mix_label} mix (speech={enhanced_speech_gain}, bg={background_gain}, music={music_gain})")
    timestamp_ms = str(int(time.time() * 1000))
    resp = requests.post(
        f'{ADOBE_API_BASE}/api/v1/enhance_speech_tracks/{track_id}/exports',
        headers={**headers, 'content-type': 'application/json'},
        params={'time': timestamp_ms},
        json={
            'enhanced_speech_gain': enhanced_speech_gain,
            'isolated_speech_gain': isolated_speech_gain,
            'background_gain': background_gain,
            'music_gain': music_gain,
            'enhancement_enabled': True,
            'track_component': 'full',
        },
    )
    if resp.status_code not in (200, 201):
        raise Exception(f'Adobe export creation failed: {resp.status_code} - {resp.text[:200]}')
    export_data = resp.json()
    export_id = export_data.get('id')
    if not export_id:
        raise Exception(f'Adobe export response missing id: {export_data}')
    print(f"Adobe Enhance: export created with {mix_label} mix, export_id={export_id}")

    # Step 6: Poll export for download URL (max 180 attempts x 5s = 15 min)
    download_url = None
    for attempt in range(180):
        time.sleep(5)
        timestamp_ms = str(int(time.time() * 1000))
        resp = requests.get(
            f'{ADOBE_API_BASE}/api/v1/enhance_speech_tracks/{track_id}/exports/{export_id}',
            headers=headers,
            params={'time': timestamp_ms},
        )
        if resp.status_code == 200:
            data = resp.json()
            url = data.get('url') or data.get('download_url')
            if url:
                download_url = url.replace('\u0026', '&')
                break
        elif resp.status_code == 204:
            if attempt % 6 == 0:
                print(f"Adobe Enhance: export rendering... ({attempt * 5}s elapsed)")
            continue
        else:
            raise Exception(f'Adobe export poll failed: {resp.status_code} - {resp.text[:200]}')
    if not download_url:
        raise Exception('Adobe export timed out after 15 minutes')
    print(f"Adobe Enhance: export ready, downloading")

    # Step 7: Download exported file
    resp = requests.get(download_url, stream=True)
    if resp.status_code != 200:
        raise Exception(f'Adobe export download failed: {resp.status_code}')

    enhanced_path = audio_path.rsplit('.', 1)[0] + '_adobe_enhanced.wav'
    with open(enhanced_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    enhanced_size = os.path.getsize(enhanced_path)
    print(f"Adobe Enhance: saved {enhanced_path} ({enhanced_size // 1024}KB)")
    _mark_token_used()
    return enhanced_path


# ============================================================================
# EDIT PIPELINE — two-phase background jobs
# ============================================================================

def _run_edit_job(job_id, audio_path, cuts_ms, transcript_id=None,
                  intro_path=None, outro_path=None, enhance_mix=None):
    """
    Background thread: apply cuts, optionally merge intro/outro, then either
    auto-enhance or stop for manual enhance.
    Pipeline: cuts → merge (if intro/outro) → Adobe enhance → finalize → completed
    Without token: cuts → merge → cuts_completed (user enhances manually)
    """
    current_step = 'cutting'
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

        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'cutting'
        wav_path = apply_audio_edits(audio_path, cuts_ms, words=words)

        # Merge intro/outro if provided
        if intro_path or outro_path:
            current_step = 'merging'
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'merging'
            wav_path = _concat_with_crossfade(wav_path, intro_path=intro_path, outro_path=outro_path)

        # If no Adobe token, stop here for manual enhancement
        if not get_adobe_token():
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'cuts_completed'
                _edit_jobs[job_id]['cuts_path'] = wav_path
            print(f"Cuts job {job_id} complete (manual enhance mode): {wav_path}")
            return

        # Auto-enhance with Adobe Enhance Speech
        current_step = 'enhancing'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'enhancing'
        enhanced_path = process_with_adobe_enhance(wav_path, enhance_mix=enhance_mix)

        # Finalize: stereo + peak-normalize + MP3
        current_step = 'finalizing'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'finalizing'
        _finalize_audio(job_id, enhanced_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        with _edit_jobs_lock:
            _edit_jobs[job_id] = {
                'status': 'error',
                'error': str(e),
                'failed_step': current_step,
            }


def _finalize_audio(job_id, enhanced_path):
    """Shared finalization: convert Adobe output to MP3 without altering levels."""
    mp3_path = enhanced_path.rsplit('.', 1)[0] + '_final.mp3'

    # Minimal processing — Adobe Enhance output is already well-mastered.
    # No loudnorm (was clipping at 0dB), no forced stereo (mono is fine for speech).
    # Just encode to high-quality MP3.
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', enhanced_path,
         '-b:a', '320k',
         '-f', 'mp3', mp3_path],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise Exception(f"ffmpeg finalize failed: {result.stderr[-500:]}")

    size_kb = os.path.getsize(mp3_path) // 1024
    print(f"Finalize: {mp3_path} ({size_kb}KB, 320kbps MP3)")

    with _edit_jobs_lock:
        _edit_jobs[job_id]['status'] = 'completed'
        _edit_jobs[job_id]['path'] = mp3_path
        _edit_jobs[job_id]['is_mp3'] = True
    print(f"Finalize job {job_id} complete: {mp3_path}")


def _run_finalize_job(job_id, enhanced_path):
    """Background thread phase 2 (manual path): finalize after user re-uploads enhanced file."""
    try:
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'finalizing'
        _finalize_audio(job_id, enhanced_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'error'
            _edit_jobs[job_id]['error'] = str(e)
            _edit_jobs[job_id]['failed_step'] = 'finalizing'


# ============================================================================
# SHOW NOTES — on-demand episode summary via Claude
# ============================================================================

def generate_show_notes(transcript_data):
    """Call Claude to generate show notes from transcript."""
    print("Generating show notes with Claude...")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    utterances = transcript_data.get('utterances', [])
    utt_lines = []
    for utt in utterances[:120]:
        start = format_timestamp(utt.get('start', 0))
        speaker = utt.get('speaker', '?')
        text = utt.get('text', '')
        utt_lines.append(f"[{start}] Speaker {speaker}: {text}")
    utt_text = "\n".join(utt_lines) if utt_lines else transcript_data.get('text', '')[:6000]

    duration_ms = transcript_data.get('audio_duration', 0)
    duration_str = format_timestamp(duration_ms) if duration_ms else 'unknown'

    prompt = f"""You are a podcast producer writing show notes for an episode. Based on the transcript below, generate concise, well-structured show notes.

TRANSCRIPT:
{utt_text}

EPISODE DURATION: {duration_str}

Generate the following sections:

## Summary
2-3 sentences summarizing the episode.

## Key Topics
- Bulleted list of the main topics discussed

## Notable Quotes
Pick 2-4 standout quotes (with speaker attribution and approximate timestamp).

## Chapters
Timestamp-based chapter markers for the episode. Format each as:
- [HH:MM:SS] Chapter title

Keep the tone professional but approachable. Be concise."""

    models = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    max_retries = 3
    response = None
    for model in models:
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                print(f"Show notes generated with {model}")
                break
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 529) and attempt < max_retries - 1:
                    wait = min(10 * (2 ** attempt), 30)
                    print(f"{model} API {e.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                elif e.status_code in (429, 529):
                    break
                else:
                    raise
        if response is not None:
            break
    if response is None:
        raise Exception("Claude API failed — all models overloaded")

    text_blocks = [b.text for b in response.content if hasattr(b, 'text')]
    return "\n".join(text_blocks)


def _run_show_notes_job(job_id, transcript_id):
    """Background thread: fetch transcript and generate show notes."""
    try:
        transcript_data = get_transcription(transcript_id)
        if transcript_data.get('status') != 'completed':
            raise Exception(f"Transcript not ready (status: {transcript_data.get('status')})")
        notes = generate_show_notes(transcript_data)
        with _jobs_lock:
            _jobs[job_id] = {'status': 'completed', 'result': {'show_notes': notes}}
        print(f"Show notes job {job_id} complete")
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _jobs_lock:
            _jobs[job_id] = {'status': 'error', 'error': str(e)}


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
        "version": "8.0.0",
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


@app.route('/api/upload-multitrack', methods=['POST', 'OPTIONS'])
def upload_multitrack():
    """Upload multiple tracks, start async per-track enhance + combine + transcription."""
    if request.method == 'OPTIONS':
        return '', 204

    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500

    # Collect track files, labels, and per-track enhance mixes from form data
    import json as _json
    track_paths = []
    labels = []
    enhance_mixes = []
    i = 0
    while f'track_{i}' in request.files:
        f = request.files[f'track_{i}']
        if f.filename == '':
            i += 1
            continue
        if not allowed_file(f.filename):
            for p in track_paths:
                if os.path.exists(p):
                    os.remove(p)
            return jsonify({"error": f"File type not allowed for {f.filename}. Use MP3, WAV, M4A, AAC, or OGG"}), 400
        ext = f.filename.rsplit('.', 1)[1].lower() if '.' in f.filename else 'wav'
        combine_id = str(uuid.uuid4())[:8]
        path = f'/tmp/{combine_id}_track{i}.{ext}'
        f.save(path)
        track_paths.append(path)
        labels.append(request.form.get(f'label_{i}', f'Track {i + 1}'))
        mix_raw = request.form.get(f'enhance_mix_{i}')
        if mix_raw:
            try:
                enhance_mixes.append(_json.loads(mix_raw))
            except Exception:
                enhance_mixes.append(None)
        else:
            enhance_mixes.append(None)
        i += 1

    if len(track_paths) < 2:
        for p in track_paths:
            if os.path.exists(p):
                os.remove(p)
        return jsonify({"error": "At least 2 tracks are required"}), 400

    job_id = str(uuid.uuid4())
    with _edit_jobs_lock:
        _edit_jobs[job_id] = {'status': 'pending'}

    print(f"Multi-track upload: {len(track_paths)} tracks, job {job_id}")

    threading.Thread(
        target=_run_multitrack_job,
        args=(job_id, track_paths, labels, enhance_mixes),
        daemon=True,
    ).start()

    return jsonify({"success": True, "job_id": job_id})


@app.route('/api/multitrack-status/<job_id>', methods=['GET'])
def multitrack_status(job_id):
    """Poll multitrack job progress. Returns transcript_id when transcription starts."""
    with _edit_jobs_lock:
        job = _edit_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job['status'] == 'error':
        return jsonify({"error": job['error']}), 500
    result = {"status": job['status']}
    if 'progress' in job:
        result['progress'] = job['progress']
    if 'transcript_id' in job:
        result['transcript_id'] = job['transcript_id']
    if 'alignment' in job:
        result['alignment'] = job['alignment']
    return jsonify(result)


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


def _run_analysis_job(job_id, transcript_data, filename, preset_name, transcript_id, custom_instructions, is_multitrack=False):
    """Background thread: auto-detect preset if needed, then run Claude analysis."""
    try:
        detected_preset = None
        detection_metrics = None

        # Auto-detect preset from audio analysis
        if preset_name == 'auto':
            with _jobs_lock:
                _jobs[job_id]['status'] = 'detecting'
            audio_path = None
            for ext in ALLOWED_EXTENSIONS:
                path = f"/tmp/{transcript_id}.{ext}"
                if os.path.exists(path):
                    audio_path = path
                    break
            if audio_path:
                detected_preset, detection_metrics = _analyze_audio(audio_path, transcript_data)
                preset_cfg = _get_preset(detected_preset)
            else:
                print("Auto-detect: audio file not found, falling back to 'zoom'")
                detected_preset = 'zoom'
                preset_cfg = _get_preset('zoom')
        else:
            preset_cfg = _get_preset(preset_name)

        with _jobs_lock:
            _jobs[job_id]['status'] = 'analyzing'

        edit_analysis = analyze_transcript_with_claude(transcript_data, preset_cfg, custom_instructions, is_multitrack=is_multitrack)
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
        if detected_preset:
            result['detected_preset'] = detected_preset
        if detection_metrics:
            result['detection_metrics'] = detection_metrics

        with _jobs_lock:
            _jobs[job_id] = {'status': 'completed', 'result': result}
        print(f"Analysis job {job_id} complete: {cuts_count} cuts")

    except anthropic.APIStatusError as e:
        print(f"Claude API error: status={e.status_code}, message={e.message}")
        retryable = e.status_code in (429, 529)
        err = f"Claude API error ({e.status_code}): {e.message}"
        with _jobs_lock:
            _jobs[job_id] = {'status': 'error', 'error': err, 'retryable': retryable}
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
        preset_name = data.get('preset', 'auto')
        custom_instructions = data.get('customInstructions', '')
        is_multitrack = data.get('isMultitrack', False)

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
            args=(job_id, transcript_data, filename, preset_name, transcript_id, custom_instructions, is_multitrack),
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


@app.route('/api/show-notes', methods=['POST', 'OPTIONS'])
def show_notes():
    """Generate show notes for a completed transcript."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        if not CLAUDE_API_KEY:
            return jsonify({"error": "CLAUDE_API_KEY not configured"}), 500
        data = request.json
        transcript_id = data.get('transcript_id')
        if not transcript_id:
            return jsonify({"error": "No transcript_id provided"}), 400

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {'status': 'pending'}

        threading.Thread(
            target=_run_show_notes_job,
            args=(job_id, transcript_id),
            daemon=True,
        ).start()

        return jsonify({"success": True, "job_id": job_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/show-notes-status/<job_id>', methods=['GET'])
def show_notes_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job['status'] == 'completed':
        return jsonify({"status": "completed", "show_notes": job['result']['show_notes']})
    if job['status'] == 'error':
        return jsonify({"error": job['error']}), 500
    return jsonify({"status": job['status']})


@app.route('/api/edit-audio', methods=['POST', 'OPTIONS'])
def edit_audio():
    """Step 3: Start async audio editing job (cuts, optional intro/outro merge)."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        # Accept both JSON and multipart form data for backwards compatibility
        import json as _json
        if request.content_type and 'multipart' in request.content_type:
            transcript_id = request.form.get('transcript_id')
            cuts_raw = request.form.get('cuts', '[]')
            cuts = _json.loads(cuts_raw)
        else:
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
            return jsonify({"error": "Audio file not found. Please re-upload."}), 404

        cuts_ms = [
            (c['start_ms'], c['end_ms'])
            for c in cuts
            if 'start_ms' in c and 'end_ms' in c
        ]

        job_id = str(uuid.uuid4())

        # Save optional intro/outro files
        intro_path = None
        outro_path = None
        if request.files:
            if 'intro' in request.files and request.files['intro'].filename:
                intro_file = request.files['intro']
                intro_ext = intro_file.filename.rsplit('.', 1)[1].lower() if '.' in intro_file.filename else 'wav'
                intro_path = f"/tmp/{job_id}_intro.{intro_ext}"
                intro_file.save(intro_path)
                print(f"Intro saved: {intro_path} ({os.path.getsize(intro_path) // 1024}KB)")
            if 'outro' in request.files and request.files['outro'].filename:
                outro_file = request.files['outro']
                outro_ext = outro_file.filename.rsplit('.', 1)[1].lower() if '.' in outro_file.filename else 'wav'
                outro_path = f"/tmp/{job_id}_outro.{outro_ext}"
                outro_file.save(outro_path)
                print(f"Outro saved: {outro_path} ({os.path.getsize(outro_path) // 1024}KB)")

        # Parse optional Adobe enhance mix ratios and multitrack flag
        enhance_mix = None
        is_multitrack = False
        if request.content_type and 'multipart' in request.content_type:
            enhance_mix_raw = request.form.get('enhance_mix')
            is_multitrack = request.form.get('is_multitrack') == 'true'
        else:
            enhance_mix_raw = data.get('enhance_mix')
            is_multitrack = data.get('is_multitrack', False)
        if enhance_mix_raw:
            try:
                enhance_mix = _json.loads(enhance_mix_raw) if isinstance(enhance_mix_raw, str) else enhance_mix_raw
            except Exception:
                pass

        print(f"Edit job: {len(cuts_ms)} cuts, intro={'yes' if intro_path else 'no'}, outro={'yes' if outro_path else 'no'}, mix={enhance_mix}, is_multitrack={is_multitrack}")

        with _edit_jobs_lock:
            _edit_jobs[job_id] = {'status': 'pending'}

        if is_multitrack:
            # Multi-track: per-track cuts → per-track enhance → combine
            if not get_adobe_token():
                return jsonify({"error": "ADOBE_ENHANCE_TOKEN is required for multi-track editing"}), 400
            threading.Thread(
                target=_run_multitrack_edit_job,
                args=(job_id, audio_path, cuts_ms, transcript_id),
                kwargs={'intro_path': intro_path, 'outro_path': outro_path},
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=_run_edit_job,
                args=(job_id, audio_path, cuts_ms, transcript_id),
                kwargs={'intro_path': intro_path, 'outro_path': outro_path, 'enhance_mix': enhance_mix},
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
    if job['status'] == 'cuts_completed':
        return jsonify({"status": "cuts_completed"})
    if job['status'] == 'error':
        return jsonify({"error": job['error'], "failed_step": job.get('failed_step')}), 500
    # merging, enhancing, enhancing_tracks, combining, finalizing, cutting, pending
    result = {"status": job['status']}
    if 'progress' in job:
        result['progress'] = job['progress']
    return jsonify(result)


@app.route('/api/edit-audio-download/<job_id>', methods=['GET'])
def edit_audio_download(job_id):
    with _edit_jobs_lock:
        job = _edit_jobs.get(job_id)
    if not job:
        return jsonify({"error": "File not ready"}), 404
    if job['status'] == 'cuts_completed':
        return send_file(job['cuts_path'], as_attachment=True, download_name='edited_podcast.wav', mimetype='audio/wav')
    if job['status'] == 'completed':
        return send_file(job['path'], as_attachment=True, download_name='edited_podcast.mp3', mimetype='audio/mpeg')
    return jsonify({"error": "File not ready"}), 404


@app.route('/api/upload-enhanced', methods=['POST', 'OPTIONS'])
def upload_enhanced():
    """Step 4: Receive enhanced audio file, kick off finalization."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        job_id = request.form.get('job_id')
        if not job_id:
            return jsonify({"error": "No job_id provided"}), 400

        with _edit_jobs_lock:
            job = _edit_jobs.get(job_id)
        if not job or job['status'] != 'cuts_completed':
            return jsonify({"error": "Job not found or cuts not completed yet"}), 400

        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        enhanced_path = f"/tmp/{job_id}_enhanced.wav"
        file.save(enhanced_path)
        print(f"Enhanced file saved: {enhanced_path} ({os.path.getsize(enhanced_path) // 1024}KB)")

        threading.Thread(
            target=_run_finalize_job,
            args=(job_id, enhanced_path),
            daemon=True,
        ).start()

        return jsonify({"success": True, "job_id": job_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
        "adobe_enhance_configured": bool(get_adobe_token()),
    })


@app.route('/api/set-adobe-token', methods=['POST', 'OPTIONS'])
def set_adobe_token_route():
    """Receive a fresh Adobe Bearer token from the Chrome extension."""
    if request.method == 'OPTIONS':
        return '', 204
    data = request.json or {}
    token = data.get('token', '').strip()
    if not token:
        return jsonify({"error": "No token provided"}), 400
    # Strip "Bearer " prefix if the extension sent it with the prefix
    if token.startswith('Bearer '):
        token = token[7:]
    set_adobe_token(token)
    print(f"Adobe token updated via API ({len(token)} chars)")
    return jsonify({"success": True})


@app.route('/api/token-status', methods=['GET'])
def token_status():
    """Extension polls this to know when to refresh the Adobe tab."""
    with _adobe_token_lock:
        return jsonify({"refresh_needed": _adobe_token_refresh_needed})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
