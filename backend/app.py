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


# Google credentials — built at runtime from GOOGLE_REFRESH_TOKEN env var.
# The client ID and secret are the default Google Cloud CLI credentials (public).
_GOOGLE_REFRESH_TOKEN = os.environ.get('GOOGLE_REFRESH_TOKEN')

if not ASSEMBLYAI_API_KEY or not CLAUDE_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY and CLAUDE_API_KEY are required")
if _adobe_token:
    print("Adobe Enhance Speech: configured")
else:
    print("WARNING: ADOBE_ENHANCE_TOKEN not set — audio editing will fail")

# In-memory job stores. Gunicorn must use threads (not processes) so these are shared.
_jobs: dict = {}
_jobs_lock = threading.Lock()
_edit_jobs: dict = {}
_edit_jobs_lock = threading.Lock()
# Multi-track metadata: tracks raw file paths for cleanup after editing
_multitrack_meta: dict = {}  # transcript_id → {track_paths, ...}
_multitrack_meta_lock = threading.Lock()
# Pre-detection cache: store results from analysis so audio editing doesn't re-run
_predetection_cache: dict = {}  # transcript_id → {stumbles, meta, stutters, fillers, filler_pct}
_predetection_cache_lock = threading.Lock()


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
    resp_data = resp.json()
    upload_url = resp_data.get("upload_url")
    if not upload_url:
        raise Exception(f"AssemblyAI upload response missing 'upload_url': {str(resp_data)[:200]}")
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
            "disfluencies": True,  # Keep filler words (um, uh, etc.) in transcript
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

    # Speaker count from utterances — normalize labels to strings
    utterances = transcript_data.get('utterances', [])
    speakers = set(str(u.get('speaker')) for u in utterances if u.get('speaker') is not None)
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

    # Transcription confidence — AssemblyAI returns 0.0-1.0 scale
    confidence = transcript_data.get('confidence', 0) or 0
    if isinstance(confidence, (int, float)) and confidence > 1:
        confidence = confidence / 100.0  # normalize if 0-100 scale

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
    'umm', 'uhh', 'erm', 'er', 'mmm', 'ahh', 'ehh',
}

HEDGING_PHRASES = {
    'you know', 'i mean', 'kind of', 'sort of', 'i think', 'i guess',
}


def _find_fillers(words):
    """Scan word list for filler words; classify each as 'disruptive' or 'natural'.
    - disruptive: mid-sentence, mid-clause, clustered with other fillers — remove these
    - natural: at sentence boundaries, natural pause points, standalone — keep these"""
    found = []
    i = 0
    while i < len(words):
        w = words[i]
        tok = w.get('text', '').lower().strip('.,!?;:')
        filler_text = None
        filler_end_idx = i

        if i + 1 < len(words):
            two = tok + ' ' + words[i + 1].get('text', '').lower().strip('.,!?;:')
            if two in FILLER_WORDS:
                filler_text = two
                filler_end_idx = i + 1

        if filler_text is None and tok in FILLER_WORDS:
            filler_text = tok
            filler_end_idx = i

        if filler_text:
            # Classify: is this filler disruptive or natural?
            # Default is disruptive — only mark as natural if there's strong
            # evidence the filler serves a purpose (real sentence boundary +
            # long pause, indicating a deliberate thinking sound).
            context = 'disruptive'

            # Only sentence-ENDING punctuation counts (not commas — ASR adds
            # commas everywhere). A filler after "really." or "right?" is natural
            # thinking. A filler after "I think," is mid-sentence and disruptive.
            at_sentence_boundary = False
            if i > 0:
                prev_text = words[i - 1].get('text', '').rstrip()
                if prev_text.endswith(('.', '?', '!')):
                    at_sentence_boundary = True

            # Check for a meaningful pause (>500ms) — indicates deliberate thought
            has_long_pause = False
            if i > 0:
                gap_before = w.get('start', 0) - words[i - 1].get('end', 0)
                if gap_before > 500:
                    has_long_pause = True
            if filler_end_idx + 1 < len(words):
                gap_after = words[filler_end_idx + 1].get('start', 0) - words[filler_end_idx].get('end', 0)
                if gap_after > 500:
                    has_long_pause = True

            # Natural = at a real sentence boundary AND has a long pause.
            # This is a deliberate "um..." while thinking between thoughts.
            if at_sentence_boundary and has_long_pause:
                context = 'natural'

            # Clustered fillers (another filler within 3 words) = always disruptive
            for offset in range(max(0, i - 3), min(len(words), filler_end_idx + 4)):
                if offset == i or offset == filler_end_idx:
                    continue
                nearby = words[offset].get('text', '').lower().strip('.,!?;:')
                if nearby in FILLER_WORDS:
                    context = 'disruptive'
                    break

            found.append({
                'text': filler_text,
                'start_ms': w.get('start', 0),
                'end_ms': words[filler_end_idx].get('end', 0),
                'speaker': w.get('speaker', '?'),
                'context': context,
            })
            i = filler_end_idx + 1
            continue
        i += 1
    return found


def _find_stutters(words):
    """Detect partial-word stutters: a truncated word followed by the complete word.
    E.g. 'commu' followed by 'community', or exact duplicates like 'so so'.
    Also catches cases where the transcriber garbles the partial word
    (e.g. 'comm' transcribed as 'come' before 'community') via fuzzy prefix matching."""
    found = []
    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i + 1]
        t1 = w1.get('text', '').lower().strip('.,!?;:')
        t2 = w2.get('text', '').lower().strip('.,!?;:')
        if not t1 or not t2:
            continue
        # Exact duplicate: "so so", "the the"
        if t1 == t2 and len(t1) >= 2:
            found.append({
                'partial': t1,
                'full': t2,
                'start_ms': w1.get('start', 0),
                'end_ms': w1.get('end', 0),
                'speaker': w1.get('speaker', '?'),
                'type': 'duplicate',
            })
            continue
        # Skip if first word is longer or equal — not a false start
        if len(t1) >= len(t2):
            continue
        # Exact prefix: "commu" + "community"
        if len(t1) >= 3 and t2.startswith(t1):
            found.append({
                'partial': t1,
                'full': t2,
                'start_ms': w1.get('start', 0),
                'end_ms': w1.get('end', 0),
                'speaker': w1.get('speaker', '?'),
                'type': 'partial',
            })
            continue
        # Fuzzy prefix: first 3+ chars match (catches transcription errors like
        # "come" before "community" — ASR often garbles truncated syllables)
        if len(t1) >= 3 and len(t2) >= 5:
            match_len = 0
            for c1, c2 in zip(t1, t2):
                if c1 == c2:
                    match_len += 1
                else:
                    break
            # At least 3 leading chars match and the short word is mostly matched
            if match_len >= 3 and match_len >= len(t1) - 1:
                found.append({
                    'partial': t1,
                    'full': t2,
                    'start_ms': w1.get('start', 0),
                    'end_ms': w1.get('end', 0),
                    'speaker': w1.get('speaker', '?'),
                    'type': 'fuzzy',
                })
                continue
        # Low-confidence short word before a longer word starting similarly
        # (ASR often outputs garbage for truncated syllables)
        c1 = w1.get('confidence', 1.0) or 1.0
        if isinstance(c1, (int, float)) and c1 > 1:
            c1 = c1 / 100.0  # normalize if 0-100 scale
        if c1 < 0.5 and len(t1) >= 2 and len(t1) < len(t2) and len(t2) >= 4:
            if t2[:2] == t1[:2]:
                found.append({
                    'partial': t1,
                    'full': t2,
                    'start_ms': w1.get('start', 0),
                    'end_ms': w1.get('end', 0),
                    'speaker': w1.get('speaker', '?'),
                    'type': 'low_confidence',
                })
    return found


def _find_hedging_clusters(words):
    """Detect regions where hedging phrases cluster (3+ within 15s = uncertain passage).
    These flag sections where the host is searching for words or unsure."""
    hedges = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            tok2 = (words[i].get('text', '').lower().strip('.,!?;:') + ' ' +
                    words[i + 1].get('text', '').lower().strip('.,!?;:'))
            if tok2 in HEDGING_PHRASES:
                hedges.append({
                    'text': tok2,
                    'start_ms': words[i].get('start', 0),
                    'end_ms': words[i + 1].get('end', 0),
                    'speaker': words[i].get('speaker', '?'),
                })
                i += 2
                continue
        i += 1

    # Find clusters: 3+ hedges within 15s window, same speaker
    clusters = []
    reported_ends = {}  # speaker -> last reported end_ms
    for i in range(len(hedges)):
        speaker = hedges[i]['speaker']
        # Skip if this hedge is inside an already-reported cluster
        if speaker in reported_ends and hedges[i]['start_ms'] <= reported_ends[speaker]:
            continue
        window = [hedges[i]]
        for j in range(i + 1, len(hedges)):
            if hedges[j]['speaker'] != speaker:
                continue
            if hedges[j]['start_ms'] - hedges[i]['start_ms'] > 15000:
                break
            window.append(hedges[j])
        if len(window) >= 3:
            end = window[-1]['end_ms']
            reported_ends[speaker] = end
            clusters.append({
                'start_ms': window[0]['start_ms'],
                'end_ms': end,
                'speaker': speaker,
                'phrases': [h['text'] for h in window],
                'count': len(window),
            })
    return clusters


def _find_pauses(words, min_ms=1000):
    """Return list of pauses longer than min_ms between consecutive words.
    Each pause is classified as 'emphatic' or 'uncertain' based on context:
    - emphatic: after a complete sentence, topic transition — trim gently
    - uncertain: near fillers, hedging, or restarts — trim aggressively"""
    pauses = []
    for i in range(len(words) - 1):
        gap_start = words[i].get('end', 0)
        gap_end = words[i + 1].get('start', 0)
        gap_ms = gap_end - gap_start
        if gap_ms >= min_ms:
            speaker_before = words[i].get('speaker', '?')
            speaker_after = words[i + 1].get('speaker', '?')

            # Score-based classification
            emphatic_score = 0
            uncertain_score = 0

            # Sentence boundary before pause (punctuation = complete thought)
            text_before = words[i].get('text', '')
            if text_before.rstrip().endswith(('.', '?', '!')):
                emphatic_score += 2

            # Filler words within 2 words before or after
            for offset in [i - 1, i - 2, i + 1, i + 2]:
                if 0 <= offset < len(words):
                    tok = words[offset].get('text', '').lower().strip('.,!?;:')
                    if tok in FILLER_WORDS:
                        uncertain_score += 2

            # Hedging phrases within 2 words
            for offset in range(max(0, i - 2), min(len(words) - 1, i + 3)):
                if offset + 1 < len(words):
                    tok2 = (words[offset].get('text', '').lower().strip('.,!?;:') + ' ' +
                            words[offset + 1].get('text', '').lower().strip('.,!?;:'))
                    if tok2 in HEDGING_PHRASES:
                        uncertain_score += 1

            # Word repetition after pause (restart signal)
            words_before = set()
            for offset in range(max(0, i - 2), i + 1):
                w = words[offset].get('text', '').lower().strip('.,!?;:')
                if len(w) >= 3:
                    words_before.add(w)
            for offset in range(i + 1, min(len(words), i + 4)):
                tok = words[offset].get('text', '').lower().strip('.,!?;:')
                if tok in words_before:
                    uncertain_score += 2
                    break

            pause_type = 'emphatic' if emphatic_score > uncertain_score else 'uncertain'

            pauses.append({
                'start_ms': gap_start,
                'end_ms': gap_end,
                'duration_ms': gap_ms,
                'before': words[i].get('text', ''),
                'after': words[i + 1].get('text', ''),
                'speaker_before': speaker_before,
                'speaker_after': speaker_after,
                'speaker_change': speaker_before != speaker_after,
                'pause_type': pause_type,
            })
    return pauses


def _find_stumbles(words):
    """Detect stumble regions where a speaker repeats a phrase multiple times
    trying to land on the clean version.

    Approach: find repeated 2-3 word phrases from the same speaker within 10s.
    The words between repetitions are expected to be abandoned attempts at the
    same idea — they'll contain variations of the same words, fillers, and
    false starts. Only reject if a completely unrelated topic appears between."""
    if len(words) < 6:
        return []

    found = []
    used_ranges = set()

    def clean(w):
        return w.get('text', '').lower().strip('.,!?;:\'"')

    # Build a set of all content word stems for similarity checking
    def stem(w):
        """Poor man's stemmer — first 5 chars of 4+ letter words."""
        return w[:5] if len(w) >= 4 else w

    def _fuzzy_phrase_match(phrase_a, phrase_b):
        """Check if two 3-word phrases match, allowing one word to be a mispronunciation.
        Returns True for exact matches or matches where all but one word are identical
        and the differing word shares a common prefix (>= 2 chars)."""
        if phrase_a == phrase_b:
            return True
        if len(phrase_a) != len(phrase_b):
            return False
        mismatches = []
        for k, (a, b) in enumerate(zip(phrase_a, phrase_b)):
            if a != b:
                mismatches.append((k, a, b))
        # Allow exactly one mismatch if the words are similar
        if len(mismatches) == 1:
            _, a, b = mismatches[0]
            # Share a common prefix of 2+ chars (e.g. "tier"/"ta", "recru"/"recruit")
            common = 0
            for ca, cb in zip(a, b):
                if ca == cb:
                    common += 1
                else:
                    break
            if common >= 2:
                return True
            # One word is very short (1-2 chars) AND shares at least 1 leading
            # char with the other — likely a fragment/mispronunciation.
            # e.g. "ta"/"tier" in "ta leaders"/"tier leaders"
            if (len(a) <= 2 or len(b) <= 2):
                short, long = (a, b) if len(a) <= len(b) else (b, a)
                if short and long and short[0] == long[0]:
                    return True
        return False

    function_words = {'um', 'uh', 'uhm', 'hmm', 'like', 'so', 'and', 'but', 'the',
                      'a', 'of', 'in', 'to', 'it', 'i', 'is', 'that', 'this',
                      'on', 'as', 'at', 'by', 'or', 'do', 'we', 'be', 'if',
                      'an', 'are', 'was', 'for', 'not', 'with', 'how', 'what',
                      'can', 'has', 'had', 'have', 'then', 'than', 'see', 'sort'}
    connectors = {'very', 'really', 'quite', 'certainly', 'definitely',
                  'actually', 'basically', 'probably', 'maybe', 'terms',
                  'kind', 'sort', 'thing', 'things', 'way', 'know',
                  'mean', 'think', 'going', 'saying', 'sure', 'sorry',
                  'voice', 'lot', 'bit', 'much', 'more', 'also',
                  'interesting', 'disruptive'}

    def _check_between_content(first_idx, last_idx, phrase, phrase_len,
                               has_sentence_boundary=False):
        """Check that words between two phrase occurrences are topically related.
        Connectors (very, more, really, etc.) are treated as neutral — they don't
        count as evidence of either related or unrelated content.
        If has_sentence_boundary=True, apply stricter checking (no short-gap exemption)
        because sentence boundaries indicate potentially complete thoughts."""
        phrase_stems = set(stem(w) for w in phrase if len(w) >= 3)
        between_content = []
        for k in range(first_idx + phrase_len, last_idx):
            w = clean(words[k])
            if w in function_words or len(w) < 3:
                continue
            if w in connectors:
                continue  # neutral — not evidence either way
            between_content.append(w)
        if not between_content:
            return True
        # With only 1-2 content words between phrases and NO sentence boundary,
        # there's not enough material for a meaningful new idea — trust that
        # it's a restart. But if there IS a sentence boundary (period/question
        # mark), even 2 words can carry meaning ("We're trying to augment")
        # so we must still check.
        if len(between_content) <= 2 and not has_sentence_boundary:
            return True
        unrelated = 0
        for w in between_content:
            w_stem = stem(w)
            related = (w_stem in phrase_stems or
                       any(w_stem.startswith(ps[:3]) or ps.startswith(w_stem[:3])
                           for ps in phrase_stems))
            if not related:
                unrelated += 1
        return unrelated <= len(between_content) * 0.5

    def _has_sentence_boundary(first_idx, last_idx, phrase_len):
        """Check if sentence-ending punctuation exists between phrase occurrences.
        A period/question mark/exclamation between repetitions strongly suggests
        two complete sentences rather than a stumble."""
        for k in range(first_idx + phrase_len, last_idx):
            text = words[k].get('text', '').rstrip()
            if text.endswith(('.', '?', '!')):
                return True
        return False

    def _has_other_speaker(first_idx, last_idx, speaker, phrase_len):
        """Check if another speaker talks between the phrase occurrences."""
        for k in range(first_idx + phrase_len, last_idx):
            if words[k].get('speaker', '?') != speaker:
                return True
        return False

    def _build_stumble(first_idx, last_idx, phrase, speaker, phrase_len, occurrences):
        """Build a stumble dict and mark indices as used."""
        stumble_start_ms = words[first_idx].get('start', 0)
        # Pull back cut end by 80ms to protect the first word of the clean version.
        # Without this buffer, the crossfade eats into the start of the kept word,
        # clipping consonants and making the transition sound unnatural.
        clean_start_ms = max(0, words[last_idx].get('start', 0) - 80)
        clean_end_idx = last_idx + phrase_len - 1
        clean_end_ms = words[clean_end_idx].get('end', 0)

        removed_words = [words[k].get('text', '') for k in range(first_idx, last_idx)]
        clean_words = []
        for k in range(last_idx, min(last_idx + phrase_len + 8, len(words))):
            if words[k].get('speaker', '?') != speaker:
                break
            clean_words.append(words[k].get('text', ''))

        for idx in range(first_idx, last_idx + phrase_len):
            used_ranges.add(idx)

        return {
            'stumble_start_ms': stumble_start_ms,
            'clean_start_ms': clean_start_ms,
            'clean_end_ms': clean_end_ms,
            'speaker': speaker,
            'removed_text': ' '.join(removed_words),
            'clean_text': ' '.join(clean_words),
            'phrase': ' '.join(phrase),
            'repetitions': len(occurrences),
        }

    # ── PASS 1: Consecutive 2-word exact duplicates ──
    # Catches "how it, how it", "Very cool. Very cool.", "how does that, how does that"
    # Allows up to 1-word gap between pairs (punctuation attaches to words in ASR,
    # so "how it," is one token — the next "how" starts the duplicate).
    # No function word filter — consecutive exact duplicates are always stutters
    # regardless of word type ("how it, how it" is clearly a stutter).
    for i in range(len(words) - 3):
        if i in used_ranges:
            continue
        speaker = words[i].get('speaker', '?')
        pair_a = (clean(words[i]), clean(words[i + 1]))
        if not pair_a[0] or not pair_a[1]:
            continue
        # Skip if both words are single characters (too short to be meaningful)
        if len(pair_a[0]) < 2 and len(pair_a[1]) < 2:
            continue
        # Look for the duplicate starting at i+2 or i+3 (allowing 1-word gap)
        for gap in range(0, 2):
            j = i + 2 + gap
            if j + 1 >= len(words):
                continue
            if words[j].get('speaker', '?') != speaker:
                continue
            pair_b = (clean(words[j]), clean(words[j + 1]))
            if pair_a == pair_b or _fuzzy_phrase_match(pair_a, pair_b):
                # Check time proximity (within 4s)
                t_start = words[i].get('start', 0)
                t_end = words[j + 1].get('end', 0)
                if t_end - t_start > 4000:
                    continue
                found.append(_build_stumble(i, j, pair_a, speaker, 2, [i, j]))
                break

    # ── PASS 2: 3-word phrase repetitions (original detection) ──
    # Widened to 20-word gap to catch multi-attempt restarts.
    phrase_len = 3
    for i in range(len(words) - phrase_len):
        if i in used_ranges:
            continue
        speaker = words[i].get('speaker', '?')
        phrase = tuple(clean(words[i + k]) for k in range(phrase_len))

        content_in_phrase = [w for w in phrase if w not in function_words and len(w) >= 3]
        if len(content_in_phrase) == 0:
            continue

        # Look for repetitions within 15s and 20 words, same speaker only.
        first_start = words[i].get('start', 0)
        occurrences = [i]

        for j in range(i + 1, len(words) - phrase_len + 1):
            if j in used_ranges:
                continue
            if words[j].get('speaker', '?') != speaker:
                continue
            if words[j].get('start', 0) - first_start > 15000:
                break
            if j - i > 20:
                break
            candidate = tuple(clean(words[j + k]) for k in range(phrase_len))
            if _fuzzy_phrase_match(candidate, phrase):
                occurrences.append(j)

        # Collect ALL occurrences first, then build stumble using the last one.
        # This ensures multi-attempt restarts (3+ tries) cut all the way to the
        # final clean version rather than stopping at the second attempt.
        if len(occurrences) < 2:
            continue

        first_idx = occurrences[0]
        last_idx = occurrences[-1]

        if _has_other_speaker(first_idx, last_idx, speaker, phrase_len):
            continue

        # Sentence boundary check: if there's sentence-ending punctuation between
        # the phrases, this is very likely two complete sentences, not a stumble.
        # Require the content check to pass in this case regardless of gap size.
        if _has_sentence_boundary(first_idx, last_idx, phrase_len):
            if not _check_between_content(first_idx, last_idx, phrase, phrase_len,
                                          has_sentence_boundary=True):
                continue

        # Content check for gaps >= 4 words (lowered from 10 to catch short
        # but meaningful between-content like "trying to augment")
        gap_words = last_idx - first_idx
        if gap_words >= 4 and not _check_between_content(first_idx, last_idx, phrase, phrase_len):
            continue

        # Divergence check: if the word immediately AFTER each occurrence is a
        # different content word, this is parallel structure / list, not a restart.
        # "because of the economic... because of AI" → diverges → skip
        # "the TA leaders Are... the TA leaders who" → function words → restart
        # "I think we're... I think in" → diverges → skip
        if len(occurrences) == 2:
            after_first = clean(words[first_idx + phrase_len]) if first_idx + phrase_len < len(words) else ''
            after_last = clean(words[last_idx + phrase_len]) if last_idx + phrase_len < len(words) else ''
            if (after_first and after_last and
                    after_first != after_last and
                    after_first not in function_words and after_first not in FILLER_WORDS and
                    after_last not in function_words and after_last not in FILLER_WORDS and
                    len(after_first) >= 3 and len(after_last) >= 3):
                continue

        found.append(_build_stumble(first_idx, last_idx, phrase, speaker, phrase_len, occurrences))

    # ── PASS 3: Multi-attempt restart detection ──
    # Catches patterns like: "it's certainly very disruptive in terms of, uh,
    # it's certainly disruptive. It's certainly very disruptive and..."
    # where a speaker makes 3+ attempts starting with the same 2 content words
    # but varying the rest. The 3-word pass may miss these because the full
    # 3-word phrase differs between attempts.
    #
    # Strategy: find 2-word content anchors repeated 2+ times by same speaker
    # within 15s. Then pick the last occurrence as the clean version.
    #
    # NOTE: This pass deliberately ignores used_ranges so it can detect the full
    # multi-attempt pattern even when Pass 1 already consumed some occurrences
    # as simple duplicates. The resulting wider cut will just merge with the
    # narrower Pass 1 cut harmlessly.
    for i in range(len(words) - 2):
        speaker = words[i].get('speaker', '?')
        anchor = (clean(words[i]), clean(words[i + 1]))

        # Need at least one content word in the anchor
        if all(w in function_words for w in anchor):
            continue
        # Need at least one word >= 4 chars to avoid ultra-common pairs
        if all(len(w) < 4 for w in anchor):
            continue
        # Skip if either word is a filler
        if anchor[0] in FILLER_WORDS or anchor[1] in FILLER_WORDS:
            continue

        first_start = words[i].get('start', 0)
        occurrences = [i]

        for j in range(i + 1, min(len(words) - 1, i + 25)):
            if words[j].get('speaker', '?') != speaker:
                continue
            if words[j].get('start', 0) - first_start > 15000:
                break
            cand = (clean(words[j]), clean(words[j + 1]) if j + 1 < len(words) else '')
            if cand == anchor:
                occurrences.append(j)

        # Need 3+ attempts for generic 2-word anchors to avoid false positives
        # on deliberate reuse ("focus on volume... focus on quality").
        # Allow 2 occurrences ONLY if the words immediately AFTER the anchor
        # are the same or fillers (true restart) — not different content words
        # (which indicates parallel structure).
        if len(occurrences) < 2:
            continue
        if len(occurrences) == 2:
            # Check: is this a restart or parallel structure?
            idx_a, idx_b = occurrences[0], occurrences[1]
            after_a = clean(words[idx_a + 2]) if idx_a + 2 < len(words) else ''
            after_b = clean(words[idx_b + 2]) if idx_b + 2 < len(words) else ''
            # If the words after both anchors are different content words,
            # this is parallel structure ("focus on volume... focus on quality")
            if (after_a and after_b and
                    after_a != after_b and
                    after_a not in function_words and after_a not in FILLER_WORDS and
                    after_b not in function_words and after_b not in FILLER_WORDS and
                    len(after_a) >= 3 and len(after_b) >= 3):
                continue

        first_idx = occurrences[0]
        last_idx = occurrences[-1]

        if _has_other_speaker(first_idx, last_idx, speaker, 2):
            continue

        # Content checks: for 3+ occurrences, distinguish restarts from natural
        # reuse. Key insight: in genuine restarts, each occurrence appears at a
        # restart boundary (preceded by fillers, pauses, or sentence breaks).
        # In natural reuse, the anchor is embedded mid-sentence.
        # "it's certainly... uh, it's certainly... It's certainly" → restart
        # "I think that... I think we're seeing... I think in" → embedded hedging
        if len(occurrences) >= 3:
            # Check how many occurrences (after the first) appear at restart
            # boundaries — preceded by a filler, short pause, or sentence break
            restart_count = 0
            for occ_idx in range(1, len(occurrences)):
                occ = occurrences[occ_idx]
                if occ == 0:
                    restart_count += 1
                    continue
                prev_word = clean(words[occ - 1]) if occ > 0 else ''
                prev_text = words[occ - 1].get('text', '').rstrip() if occ > 0 else ''
                # Restart boundary: previous word is filler, ends with punctuation,
                # or there's a >300ms gap before this occurrence
                is_filler = prev_word in FILLER_WORDS or prev_word in function_words
                is_sentence_break = prev_text.endswith(('.', '?', '!', ','))
                gap_before = words[occ].get('start', 0) - words[occ - 1].get('end', 0) if occ > 0 else 0
                is_pause = gap_before > 300
                if is_filler or is_sentence_break or is_pause:
                    restart_count += 1
            # If most occurrences are at restart boundaries, it's a genuine restart
            # If most are embedded mid-sentence, it's natural reuse → skip
            if restart_count < len(occurrences) - 1:
                # Most occurrences are mid-sentence → natural reuse, not restart
                continue

        if len(occurrences) == 2:
            # Sentence boundary check: two complete sentences reusing a phrase is
            # not a stumble (e.g. "our hiring process. We're trying to augment
            # our hiring process")
            if _has_sentence_boundary(first_idx, last_idx, 2):
                if not _check_between_content(first_idx, last_idx, anchor, 2,
                                              has_sentence_boundary=True):
                    continue

            # For 2-occurrence restarts, check between-content is topically related
            if not _check_between_content(first_idx, last_idx, anchor, 2):
                continue

        # Expand cut start backward: if the word(s) before the first occurrence
        # are a false start of the same phrase (e.g. "it's um, it's certainly" —
        # the leading "it's" before "um" should also be cut). Look back up to
        # 3 words for matching start word + filler/pause.
        expanded_first = first_idx
        if first_idx >= 2:
            for lookback in range(2, min(5, first_idx + 1)):
                prev_idx = first_idx - lookback
                if words[prev_idx].get('speaker', '?') != speaker:
                    break
                if clean(words[prev_idx]) == anchor[0]:
                    # Check that words between are fillers/function words
                    all_filler = True
                    for k in range(prev_idx + 1, first_idx):
                        wk = clean(words[k])
                        if wk not in FILLER_WORDS and wk not in function_words and len(wk) >= 3:
                            all_filler = False
                            break
                    if all_filler:
                        expanded_first = prev_idx
                    break

        found.append(_build_stumble(expanded_first, last_idx, anchor, speaker, 2, occurrences))

    # ── PASS 4: Sentence-level repetition detection ──
    # Catches full repeated phrases/sentences like:
    # "coming back to talk coming back to talk about AI"
    # "recruiters are going to become even more valuable, I think
    #  recruiters are going to become even more valuable"
    # Strategy: look for 4+ word sequences repeated by same speaker within 30 words.
    # IMPORTANT: Must NOT catch deliberate parallel structure like "before that I
    # spent 3 years at Citadel... before that I spent 8 years at Red Hat" — these
    # reuse the same opening but introduce different content after.
    #
    # NOTE: Like Pass 3, this pass ignores used_ranges so it can detect wider
    # patterns even when earlier passes consumed part of the repeated phrase.
    for phrase_len in [5, 4]:
        for i in range(len(words) - phrase_len):
            speaker = words[i].get('speaker', '?')
            phrase = tuple(clean(words[i + k]) for k in range(phrase_len))

            # Need at least 2 content words
            content_count = sum(1 for w in phrase if w not in function_words and len(w) >= 3)
            if content_count < 2:
                continue

            first_start = words[i].get('start', 0)

            for j in range(i + phrase_len, min(len(words) - phrase_len + 1, i + 35)):
                if words[j].get('speaker', '?') != speaker:
                    continue
                if words[j].get('start', 0) - first_start > 20000:
                    break
                candidate = tuple(clean(words[j + k]) for k in range(phrase_len))
                if candidate == phrase:
                    if _has_other_speaker(i, j, speaker, phrase_len):
                        continue
                    # Check that this is a TRUE restart, not parallel structure.
                    # Compare the words AFTER each occurrence — if they diverge
                    # immediately into different content, this is deliberate reuse
                    # (e.g. "before that I spent 3 years at Citadel... before that
                    # I spent 8 years at Red Hat"). If they continue similarly or
                    # the first attempt is abandoned (ends in filler/pause), it's a restart.
                    # SKIP this check when phrases are adjacent or nearly adjacent
                    # (j <= i + phrase_len + 2) — in that case, "after_first" is just
                    # the start of the second occurrence, not real continuation content.
                    if j > i + phrase_len + 2:
                        after_first = clean(words[i + phrase_len]) if i + phrase_len < len(words) else ''
                        after_second = clean(words[j + phrase_len]) if j + phrase_len < len(words) else ''
                        # If the words after both occurrences are different content words,
                        # this is parallel structure, not a restart
                        if (after_first and after_second and
                                after_first != after_second and
                                after_first not in function_words and
                                after_first not in FILLER_WORDS and
                                after_second not in function_words and
                                after_second not in FILLER_WORDS and
                                len(after_first) >= 3 and len(after_second) >= 3):
                            continue
                    found.append(_build_stumble(i, j, phrase, speaker, phrase_len, [i, j]))
                    break

    found.sort(key=lambda x: x['stumble_start_ms'])
    return found


def _find_meta_commentary(words):
    """Detect meta-commentary where speakers reference the recording process
    or their own performance mid-episode. These should be flagged for removal.
    E.g. 'my voice is going', 'sorry let me start again', 'I keep stumbling'."""
    META_PHRASES = [
        ('my', 'voice', 'is'),
        ('voice', 'is', 'going'),
        ('losing', 'my', 'voice'),
        ('sorry', 'about', 'that'),
        ('sorry', 'about', 'this'),
        ('let', 'me', 'start', 'again'),
        ('let', 'me', 'rephrase'),
        ('let', 'me', 'try', 'again'),
        ("i'll", 'start', 'again'),
        ("i'll", 'try', 'again'),
        ('sorry', "i'm", 'stumbling'),
        ('sorry', 'i', 'keep'),
        ('bear', 'with', 'me'),
        ('excuse', 'me'),
        ('pardon', 'me'),
        ('where', 'was', 'i'),
        ('lost', 'my', 'train'),
        ('lost', 'my', 'thought'),
        ('apologies', 'for'),
    ]

    # Single-word apologies that indicate a stumble recovery — "sorry" standalone
    # (not "sorry, our..." which is part of normal speech)
    STANDALONE_SORRY = {'sorry', 'apologies'}
    # Words that follow "sorry" in normal speech (not a stumble apology)
    SORRY_CONTINUERS = {'about', 'for', 'that', 'this', 'but', 'if', 'to', 'i'}

    found = []
    for i in range(len(words)):
        for phrase in META_PHRASES:
            plen = len(phrase)
            if i + plen > len(words):
                continue
            match = True
            for j in range(plen):
                tok = words[i + j].get('text', '').lower().strip('.,!?;:\'"')
                if tok != phrase[j]:
                    match = False
                    break
            if match:
                # Expand to capture the full meta-comment sentence
                start_idx = i
                end_idx = i + plen - 1
                # Expand forward to end of sentence (punctuation or pause >500ms)
                for k in range(end_idx + 1, min(len(words), end_idx + 10)):
                    if words[k].get('speaker', '?') != words[i].get('speaker', '?'):
                        break
                    end_idx = k
                    text = words[k].get('text', '')
                    if text.rstrip().endswith(('.', '?', '!')):
                        break
                    if k + 1 < len(words):
                        gap = words[k + 1].get('start', 0) - words[k].get('end', 0)
                        if gap > 500:
                            break

                removed_text = ' '.join(words[k].get('text', '') for k in range(start_idx, end_idx + 1))
                found.append({
                    'start_ms': words[start_idx].get('start', 0),
                    'end_ms': words[end_idx].get('end', 0),
                    'speaker': words[i].get('speaker', '?'),
                    'text': removed_text,
                    'phrase': ' '.join(phrase),
                })
                break

    # Detect standalone "sorry" / "apologies" that indicate stumble recovery
    # e.g. "...of our roles in sort sorry, our core back office..."
    for i in range(len(words)):
        tok = words[i].get('text', '').lower().strip('.,!?;:\'"')
        if tok not in STANDALONE_SORRY:
            continue
        # Check if next word is a normal continuer (e.g. "sorry about") — if so, skip
        if i + 1 < len(words):
            next_tok = words[i + 1].get('text', '').lower().strip('.,!?;:\'"')
            if next_tok in SORRY_CONTINUERS:
                continue
        # Check if already captured by phrase detection above
        already_found = False
        for f in found:
            if f['start_ms'] <= words[i].get('start', 0) <= f['end_ms']:
                already_found = True
                break
        if already_found:
            continue
        # This is a standalone sorry — expand backward to capture the stumble before it
        start_idx = i
        end_idx = i
        speaker = words[i].get('speaker', '?')
        # Look backward for the start of the stumble (up to 3s before "sorry")
        for k in range(i - 1, max(-1, i - 15), -1):
            if words[k].get('speaker', '?') != speaker:
                break
            gap = words[k + 1].get('start', 0) - words[k].get('end', 0)
            # If there's a big pause before this word, the stumble starts after it
            if gap > 800:
                start_idx = k + 1
                break
            start_idx = k
        removed_text = ' '.join(words[k].get('text', '') for k in range(start_idx, end_idx + 1))
        found.append({
            'start_ms': words[start_idx].get('start', 0),
            'end_ms': words[end_idx].get('end', 0),
            'speaker': speaker,
            'text': removed_text,
            'phrase': 'sorry (standalone)',
        })

    return found


def _find_nonverbal_gaps(words):
    """Detect gaps between words where non-speech sounds (coughs, throat clears,
    lip smacks) are likely hiding. These gaps are too long for natural pauses
    between words but aren't speaker changes.

    A gap of 500-3000ms between two words by the same speaker mid-sentence
    very likely contains audible non-verbal sounds that should be trimmed."""
    found = []
    for i in range(1, len(words)):
        prev = words[i - 1]
        curr = words[i]
        # Same speaker only
        if prev.get('speaker', '?') != curr.get('speaker', '?'):
            continue
        prev_end = prev.get('end', 0)
        curr_start = curr.get('start', 0)
        gap = curr_start - prev_end
        # Gap must be significant (>1200ms) but not so long it's a natural pause
        # between topics (>4000ms). Natural thinking pauses are 400-1000ms,
        # so only flag gaps well beyond that — these likely contain coughs.
        if gap < 1200 or gap > 4000:
            continue
        # Don't trim gaps at sentence boundaries (period/question mark before gap)
        prev_text = prev.get('text', '').rstrip()
        if prev_text.endswith(('.', '?', '!')):
            continue
        # Trim the gap: leave 100ms after previous word and 100ms before next word
        trim_start = prev_end + 100
        trim_end = curr_start - 100
        if trim_end > trim_start:
            found.append({
                'start_ms': trim_start,
                'end_ms': trim_end,
                'gap_ms': gap,
                'before_word': prev.get('text', ''),
                'after_word': curr.get('text', ''),
                'speaker': prev.get('speaker', '?'),
            })
    return found


def format_timestamp(milliseconds):
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _normalize_words(words):
    """Normalize word objects for consistent processing:
    - Ensure start/end are ints (milliseconds)
    - Ensure speaker is a string
    - Filter out words with invalid timestamps"""
    normalized = []
    for w in words:
        start = w.get('start', 0)
        end = w.get('end', 0)
        # Coerce to int, handle None
        try:
            start = int(start) if start is not None else 0
            end = int(end) if end is not None else 0
        except (ValueError, TypeError):
            continue  # skip words with unparseable timestamps
        if end < start or start < 0:
            continue  # skip invalid ranges
        speaker = w.get('speaker')
        w_copy = dict(w)
        w_copy['start'] = start
        w_copy['end'] = end
        w_copy['speaker'] = str(speaker) if speaker is not None else '?'
        normalized.append(w_copy)
    return normalized


def analyze_transcript_with_claude(transcript_data, preset_cfg, custom_instructions="", is_multitrack=False):
    """Pre-detect fillers/pauses, then ask Claude for editorial decisions."""
    print("Analyzing transcript with Claude...")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    words = _normalize_words(transcript_data.get("words", []))
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
        f'{f["start_ms"]} {f["end_ms"]} "{f["text"]}" (Speaker {f["speaker"]}) [{f.get("context", "disruptive").upper()}]'
        for f in fillers[:200]
    ]
    filler_text = "\n".join(filler_lines) if filler_lines else "None detected"

    remove_pauses = preset_cfg.get('remove_pauses', True)
    pause_min_ms = preset_cfg.get('pause_min_ms', 2000)
    pause_target_ms = preset_cfg.get('pause_target_ms', 800)
    pauses = _find_pauses(words, min_ms=pause_min_ms) if remove_pauses else []
    pause_lines = []
    for p in pauses[:100]:
        tag = p.get('pause_type', 'uncertain').upper()
        line = f'{p["start_ms"]} {p["end_ms"]} {p["duration_ms"]}ms  ("{p["before"]}" \u2192 "{p["after"]}") [{tag}]'
        if is_multitrack and p.get('speaker_change'):
            line += f'  [SPEAKER CHANGE: {p["speaker_before"]}\u2192{p["speaker_after"]}]'
        pause_lines.append(line)
    pause_text = "\n".join(pause_lines) if pause_lines else "None detected"

    stutters = _find_stutters(words)
    stutter_lines = [
        f'{s["start_ms"]} {s["end_ms"]} "{s["partial"]}" \u2192 "{s["full"]}" [{s["type"]}] (Speaker {s["speaker"]})'
        for s in stutters[:100]
    ]
    stutter_text = "\n".join(stutter_lines) if stutter_lines else "None detected"

    hedging_clusters = _find_hedging_clusters(words)
    hedging_lines = [
        f'{c["start_ms"]} {c["end_ms"]} Speaker {c["speaker"]}: {c["count"]}x hedging ({", ".join(c["phrases"])})'
        for c in hedging_clusters[:50]
    ]
    hedging_text = "\n".join(hedging_lines) if hedging_lines else "None detected"

    stumbles = _find_stumbles(words)
    stumble_lines = [
        f'{s["stumble_start_ms"]}–{s["clean_start_ms"]} Speaker {s["speaker"]}: '
        f'"{s["phrase"]}" repeated {s["repetitions"]}x — '
        f'CUT {s["stumble_start_ms"]}–{s["clean_start_ms"]} '
        f'(remove: "{s["removed_text"][:80]}") '
        f'(keep from {s["clean_start_ms"]}: "{s["clean_text"][:60]}")'
        for s in stumbles[:20]
    ]
    stumble_text = "\n".join(stumble_lines) if stumble_lines else "None detected"

    meta_comments = _find_meta_commentary(words)
    meta_lines = [
        f'{m["start_ms"]}–{m["end_ms"]} Speaker {m["speaker"]}: '
        f'"{m["text"]}" (meta-commentary, remove)'
        for m in meta_comments[:20]
    ]
    meta_text = "\n".join(meta_lines) if meta_lines else "None detected"

    print(f"Pre-detected {len(fillers)} fillers, {len(pauses)} pauses, {len(stutters)} stutters, "
          f"{len(hedging_clusters)} hedging clusters, {len(stumbles)} stumbles, {len(meta_comments)} meta-comments")

    multitrack_rules = ""
    if is_multitrack:
        multitrack_rules = (
            "\nMULTI-TRACK RULES (CRITICAL \u2014 this audio was mixed from separate speaker tracks):\n"
            "- SPEAKER TRANSITION PAUSES: When a pause occurs between two DIFFERENT speakers, do NOT trim it shorter than 1.5 seconds. "
            "Trimming speaker transitions too aggressively causes speakers to sound like they are talking over each other because each track "
            "contains background audio from the recording environment.\n"
            "- FILLER WORDS NEAR TRANSITIONS: Do NOT remove filler words that occur within 500ms of another speaker starting or stopping. "
            "These fillers overlap with the other speaker's audio on their track.\n"
            "- Be MORE CONSERVATIVE overall with pause trimming \u2014 it is far better to have a slightly longer pause than to create speaker overlap artifacts.\n"
            "- If a pause is between the SAME speaker's sentences, normal trimming rules apply.\n"
        )

    prompt = f"""You are an experienced podcast editor. Your task is to clean and polish this podcast episode while keeping a natural, human flow. The goal is clean, confident, and human \u2014 NOT over-edited.

UTTERANCE TRANSCRIPT (for context):
{utt_text}

PRE-DETECTED FILLER WORDS (format: start_ms end_ms "word" speaker):
{filler_text}

PRE-DETECTED PAUSES >{pause_min_ms}ms (format: start_ms end_ms duration before\u2192after):
{pause_text}

PRE-DETECTED STUTTERS (format: start_ms end_ms "partial" \u2192 "full" [type] speaker):
{stutter_text}

PRE-DETECTED HEDGING CLUSTERS (3+ hedging phrases within 15s \u2014 uncertain passages):
{hedging_text}

PRE-DETECTED STUMBLES (repeated phrases — speaker trying to land on the right wording):
{stumble_text}
For each stumble above: use the exact CUT range provided (stumble_start_ms to clean_start_ms). This removes all abandoned attempts and keeps the final clean version. Do NOT adjust these boundaries — they are word-level precise.

PRE-DETECTED META-COMMENTARY (speaker references recording/performance — remove these):
{meta_text}
Meta-commentary is where a speaker breaks the fourth wall to comment on the recording itself ("my voice is going", "sorry let me start again", "bear with me"). These must be removed — listeners should not hear them. Use the exact start_ms and end_ms provided.

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
Each filler above is tagged [DISRUPTIVE] or [NATURAL] based on its position in the speech:
- [DISRUPTIVE]: mid-sentence, mid-clause, or clustered with other fillers — these interrupt flow. Remove most of these.
- [NATURAL]: at sentence boundaries, after pauses, standalone thinking sounds — these are part of natural human speech rhythm. KEEP these unless they are excessive.
- Target: remove approximately {filler_pct}% of [DISRUPTIVE] fillers. Keep most [NATURAL] fillers — a thoughtful "um" at a pause point sounds human and should stay.
- "Remove a filler" means cutting ONLY the filler word itself (using its exact start_ms and end_ms). Do NOT remove surrounding words. The words before and after must remain fully intact.
- Use the EXACT start_ms and end_ms provided — never adjust these timestamps.
- Preserve ALL acronyms and industry-specific terms exactly as spoken.
- SMOOTHNESS: Only remove fillers where the surrounding audio flows naturally without them. If removing a filler would create an awkward jump or unnatural rhythm, leave it in.

PAUSE RULES:
Each pause above is tagged [EMPHATIC] or [UNCERTAIN] based on surrounding context. Treat them differently:
- [UNCERTAIN] pauses (near fillers, hedging, or restarts): Trim aggressively to ~0.5s. These are the speaker losing their thread, not dramatic effect.
- [EMPHATIC] pauses (after complete sentences, topic transitions): Trim gently to ~1.2s, or leave entirely if the pause genuinely serves the narrative. A beat after a powerful statement makes it land harder \u2014 do not flatten that.
- You may override the classification if you disagree based on the transcript context. The tags are heuristic, not gospel.
- Set start_ms = pause_start_ms + target, end_ms = pause_end_ms (target = 500 for uncertain, 1200 for emphatic).
- All pauses in the list should be addressed \u2014 either trimmed or explicitly left.
{multitrack_rules}CONTENT RULES:
- STUTTERS: Remove ALL pre-detected stutters listed above. For each stutter, cut the partial/duplicate word using its start_ms and end_ms. These are the safest type of content cut.
- STUTTERS (scan independently): Also look through the transcript for stutters the pre-detection missed. These include: exact word duplicates ("so so"), partial-word false starts where a speaker begins a word then restarts it ("comm community", "compu computer", "tic particularly"), and repeated short phrases ("I think I think"). The transcriber may garble the partial word, so look for any short word immediately before a longer word that sounds like a false start of that word. Cut the partial/duplicate.
- STUMBLES: For each PRE-DETECTED STUMBLE listed above, verify before applying: (1) read the removed text — is it genuinely abandoned/repeated phrasing? If it contains NEW ideas, names, facts, or meaningful content NOT present elsewhere, SKIP this stumble. (2) Read the resulting sentence with the cut applied — is it grammatical and natural? If not, SKIP. (3) Only if both checks pass, apply the cut using the EXACT start_ms and end_ms provided. CRITICAL FALSE POSITIVE CHECK: Parallel structures like "before that I spent 3 years at X... before that I spent 8 years at Y" or "focus on volume... focus on quality" reuse the same opening but introduce DIFFERENT content. These are NOT stumbles — they are deliberate rhetorical structure. If the content AFTER the repeated phrase is different, SKIP.
- FALSE STARTS: Only cut when a speaker clearly abandons a sentence and restarts THE SAME sentence with the SAME words. You must be very confident the restart is cleaner. If in doubt, leave both. Always make ONE continuous cut — never split into separate cuts that leave fragments between them.
  DO NOT CUT these patterns — they are natural speech, not false starts:
  - LISTS: "because of the economy, because of AI" — parallel items in a list
  - QUALIFICATION: "I think we're seeing... at least I think in the tech space" — narrowing a claim
  - PREPOSITIONAL CHAINS: "impact some of the nature of the workforce" — not a repetition
  - PARENTHETICAL ASIDES: "for our... for everyone once they're here, for our existing people" — the aside adds context
  Only cut when the SAME words are repeated with NO new information between them.
- RESTATED THOUGHTS / IMMEDIATE REPETITIONS: Cut when a speaker says 3+ words then IMMEDIATELY repeats the same words ("coming back to talk coming back to talk", "recruiters are going to become even more valuable, I think recruiters are going to become even more valuable"). Keep the second (cleaner) version. Also cut when a speaker rephrases the exact same idea immediately — the second version must fully replace the first with zero loss of meaning. Make ONE continuous cut covering all abandoned versions, ending right at the start of the kept version.
- META-COMMENTARY: Remove ALL pre-detected meta-commentary listed above using the exact timestamps. Also scan the transcript for any meta-commentary the pre-detection missed — moments where a speaker comments on the recording itself, their own performance, or breaks from the conversation topic to address a technical issue. These break the listener's immersion and must be removed. IMPORTANT: Each meta-commentary must be its own separate Content Cut with its own start_ms and end_ms. Do NOT merge meta-commentary with nearby stumble cuts into one giant cut — keep them as separate decisions so the system can process them independently.
- HEDGING CLUSTERS: In regions flagged as hedging clusters above, you may remove 1-2 individual hedging phrases ("you know", "I mean", "kind of") — cut ONLY those exact words, not the sentence around them. Do NOT strip all hedging \u2014 some is natural.
- REPEATED POINTS: When a speaker makes the exact same point twice in immediate succession, keep the stronger version. This should be rare — only when the repetition is truly redundant.
- NEVER remove an entire sentence or clause just because it contains filler words like "kind of", "sort of", "you know". Remove the filler words themselves if needed, but KEEP the surrounding sentence — it carries meaning and context. A sentence with a filler removed is always better than a sentence deleted entirely.
- Do NOT make speculative content cuts. Only cut content you are 90%+ confident should be removed.
- Do NOT rewrite or paraphrase content. Do NOT change the meaning or tone of the speaker.
- All content cuts MUST start at the beginning of a word (use the word's start_ms) and end at the end of a word (use the word's end_ms). NEVER cut mid-word.
- SPEAKER BOUNDARY: Content cuts must NEVER cross speaker boundaries. A cut that starts in Speaker A's words must end in Speaker A's words. NEVER remove the end of one speaker's sentence to cut into a stumble by another speaker. Check the speaker labels — if your cut region contains words from more than one speaker, shrink it to only cover the stumbling speaker's words.
- SAFETY CHECK: Before finalizing any Content Cut, mentally read the sentence with the cut applied. If the remaining words do not form a complete, grammatical sentence, do NOT make the cut. Also verify that no other speaker's words are being removed.
- SCOPE CHECK: If a Content Cut spans more than 8 seconds, you are almost certainly cutting too much. Re-examine whether you are removing an entire thought that should be kept. For stutters, the cut should be under 2 seconds. For false starts and restated thoughts, the cut can be longer (up to 8 seconds) if the speaker truly repeated themselves multiple times before landing on the clean version — but make it ONE continuous cut, not multiple separate cuts.

PRESERVE RULES (do NOT cut these):
- TRANSITIONAL PHRASES that connect topics or introduce new points, even if they seem tangential. E.g. "I think the other thing I'd flag which maybe doesn't fall under compliance but it's part of it" \u2014 these bridge ideas and must be kept.
- AGREEMENT MARKERS between speakers like "yeah absolutely", "yeah exactly", "absolutely" \u2014 these show active engagement and are part of natural conversation flow.
- SHORT RESPONSES between speakers like "yeah" or "right" that acknowledge the other speaker \u2014 these maintain conversational rhythm and show listening. Only cut if they overlap with the other speaker's words.
- SPEAKER TRANSITIONS where one person hands off to another \u2014 keep the social glue that makes dialogue sound natural.

STRUCTURAL RULES:
- PRE-RECORDING CHAT: Scan the first 90 seconds for pre-recording logistics. These come in two forms:
  (a) EXPLICIT logistics: "okay recording now", "we are now recording", "are we recording?", "let me hit record", mic checks, countdown cues.
  (b) READINESS cues: "are you all set?", "all set", "ready?", "ready to go?", "shall we start?", "okay I'll start with this", "let's get into it", "let's go", "here we go". These are the host confirming they are about to begin — everything before them (and including them) is pre-chat.
  After these cues, the host typically leaves a brief pause before starting. Look for that pause, then the EPISODE OPENER — the first moment the host addresses the audience or guest in an on-air voice. Common patterns:
  - Greeting + guest name: "Hi, John", "Hello, Sarah", "Welcome, David"
  - Audience greeting: "welcome back", "welcome to", "hey everyone", "hello and welcome", "good morning"
  - Topic launch: "so today we're going to", "today I'm joined by", "in this episode"
  EVERYTHING before the episode opener is pre-chat and MUST be removed. CRITICAL STEPS: (1) Find the readiness cue or logistics phrase. (2) Find the episode opener AFTER that cue. (3) Set your cut start_ms to 0 and cut end_ms to the opener word's start_ms. The system will add a tiny buffer automatically — do NOT subtract manually. The opener word must start cleanly and completely. (4) If there is no pre-recording chat or readiness cue, do NOT make this cut. When in doubt, keep more — it is far worse to clip the opener than to leave in pre-chat.
- LAST WORD PROTECTION: When making any cut near the end of the episode, ensure the final word of real content is fully preserved. Never set a cut's start_ms within the last spoken word — use the word's end_ms as the earliest allowed cut point.
- END-OF-EPISODE PROTECTION: In the last 25% of the episode, be EXTREMELY conservative with Content Cuts. Speakers often deliver concluding thoughts, summaries, or sign-offs that may sound like restated ideas but are actually the intended wrap-up. Do NOT cut restated thoughts or verbal stumbles in the final quarter — only remove pre-detected fillers, stutters, and clearly abandoned false starts (< 3 seconds). If a speaker repeats an idea near the end, they are likely emphasizing it deliberately.
- POST-INTERVIEW CHAT: If there is chat after the episode has clearly concluded ("okay I'll stop recording", "that was great", wrap-up logistics), mark it for removal. Do NOT confuse a speaker's concluding remarks or sign-off with post-interview chat — if they are still addressing the audience or making a point, it is content.
- Preserve the full interview/episode content itself.

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
            raw_decisions = block.input.get("decisions", [])
            if not isinstance(raw_decisions, list):
                raw_decisions = []
            # Validate and coerce each decision's timestamps
            edit_decisions = []
            for d in raw_decisions:
                if not isinstance(d, dict):
                    continue
                s, e = d.get('start_ms'), d.get('end_ms')
                if s is not None and e is not None:
                    try:
                        d['start_ms'] = int(s)
                        d['end_ms'] = int(e)
                        if d['end_ms'] <= d['start_ms']:
                            continue  # skip invalid ranges
                    except (ValueError, TypeError):
                        continue  # skip unparseable timestamps
                edit_decisions.append(d)
            break

    if edit_decisions is None:
        raise Exception("Claude did not return tool call \u2014 unexpected response format")

    print(f"Generated {len(edit_decisions)} edit decisions")

    # ---- REVIEW PASS: build post-edit transcript and check for missed issues ----
    # Collect all cut ranges from the initial decisions
    initial_cuts = []
    for d in edit_decisions:
        s = d.get('start_ms')
        e = d.get('end_ms')
        if s is not None and e is not None:
            initial_cuts.append((int(s), int(e)))

    # Also include injected stumble/meta/stutter cuts (using already-computed data)
    for stum in stumbles:
        initial_cuts.append((int(stum['stumble_start_ms']), int(stum['clean_start_ms'])))
    for meta in meta_comments:
        initial_cuts.append((int(meta['start_ms']), int(meta['end_ms'])))
    for stut in stutters:
        initial_cuts.append((int(stut['start_ms']), int(stut['end_ms'])))

    # Sort and merge
    initial_cuts.sort()
    merged_cuts = []
    for s, e in initial_cuts:
        if merged_cuts and s <= merged_cuts[-1][1] + 150:
            merged_cuts[-1] = (merged_cuts[-1][0], max(merged_cuts[-1][1], e))
        else:
            merged_cuts.append((s, e))

    # Build the post-edit word list (words that survive the cuts)
    surviving_words = []
    for w in words:
        ws = w.get('start', 0)
        we = w.get('end', 0)
        cut = False
        for cs, ce in merged_cuts:
            if ws >= cs and we <= ce:
                cut = True
                break
        if not cut:
            surviving_words.append(w)

    # Build post-edit transcript with word-level timestamps and gap markers
    # Gap markers show where non-speech sounds (coughs, throat clears) might be hiding
    review_lines = []
    current_speaker = None
    current_line = []
    prev_end = 0
    for w in surviving_words:
        speaker = w.get('speaker', '?')
        ws = w.get('start', 0)
        we = w.get('end', 0)
        gap = ws - prev_end

        if speaker != current_speaker:
            if current_line:
                review_lines.append(f"Speaker {current_speaker}: {' '.join(current_line)}")
            current_speaker = speaker
            current_line = []

        # Mark significant gaps (>500ms) — could contain coughs or non-speech sounds
        if gap > 500 and prev_end > 0:
            current_line.append(f"[GAP {gap}ms]")

        text = w.get('text', '')
        current_line.append(f"{text}({ws})")
        prev_end = we
    if current_line:
        review_lines.append(f"Speaker {current_speaker}: {' '.join(current_line)}")

    review_transcript = "\n".join(review_lines)

    print("Running review pass on post-edit transcript...")
    review_prompt = f"""You are a podcast editor doing a FINAL QUALITY CHECK on an already-edited transcript. The initial edit has already removed fillers, stutters, and most stumbles. Your job is to find ONLY clear, obvious disfluencies that were missed.

Below is the POST-EDIT transcript — this is what the listener will hear. Each word has its original timestamp in milliseconds: word(12345). [GAP Xms] markers show gaps between words where non-speech sounds may be.

{review_transcript}

LOOK FOR ONLY THESE SPECIFIC ISSUES:

1. MISPRONOUNCED WORDS: Speaker says a word wrong then immediately corrects it ("tier leaders... TA leaders"). Cut ONLY the mispronounced word(s), keep the correction. CRITICAL: The mispronounced word must be in the SAME grammatical position as the correction — i.e., it replaces the same slot in the sentence. Do NOT cut a word from an earlier phrase just because it sounds vaguely similar to a later word. For example, in "the first thing I think about in this is TA leaders", "thing" is NOT a mispronunciation of "TA" — it belongs to a different phrase ("the first thing").

2. META-COMMENTARY: Speaker says "sorry", "excuse me", "my voice is going" etc. Cut the meta-commentary only.

3. COUGHS/THROAT CLEARS: Look for [GAP] markers that might contain audible non-speech sounds between words. If a gap is followed by a restart of the same phrase, cut the gap region.

4. IMMEDIATE RESTARTS: Speaker says 3+ words then immediately restarts with the SAME words ("coming back to talk coming back to talk about AI", "the TA leaders are listening to the TA leaders who are listening"). Cut the first attempt, keep the second (cleaner) version.

5. SENTENCE REPETITION: Speaker says a full clause/sentence, then repeats essentially the same sentence immediately ("recruiters are going to become even more valuable, I think recruiters are going to become even more valuable"). Cut the first version, keep the second.

CRITICAL RULES — DO NOT VIOLATE:
- Do NOT cut content where a speaker is elaborating or developing a NEW thought.
- Natural thematic repetition across paragraphs is fine — only cut IMMEDIATE back-to-back repetition of the same words.
- Do NOT cut abandoned thoughts that contain meaningful NEW content.
- Do NOT cut sentences where the speaker completes a thought then continues. "It's going to be a very different work. It's probably going to be less around sourcing" — this is a completed statement followed by elaboration, NOT a restart.
- Do NOT cut sentences with a long pause in them — pauses are natural.
- Maximum cut length: 8 seconds. If you think something longer needs cutting, you're probably cutting content.
- Only make cuts where you are VERY confident (>90%) the result sounds better.
- SAFETY CHECK: Before making any cut, mentally read the sentence with the cut applied. If the remaining words do not form a complete, grammatical sentence, do NOT make the cut.
- When in doubt, DO NOT CUT. It is better to leave a minor disfluency than to remove meaningful content.
- If you find no clear issues, return a single Note saying "Review pass: no additional edits needed."

Call the submit_edit_decisions tool with your decisions."""

    review_message = None
    for attempt in range(3):
        try:
            review_message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_edit_decisions"},
                messages=[{"role": "user", "content": review_prompt}],
            )
            break
        except Exception as e:
            if attempt < 2:
                print(f"Review pass attempt {attempt + 1} failed: {e}")
            else:
                print(f"Review pass failed after 3 attempts: {e}")

    review_decisions = []
    if review_message:
        for block in review_message.content:
            if block.type == "tool_use" and block.name == "submit_edit_decisions":
                review_decisions = block.input.get("decisions", [])
                break

    # Review pass: high-confidence short cuts (>=93%, <=3s) are applied.
    # Everything else becomes advisory Notes to prevent content removal.
    applied_review = 0
    advisory_review = 0
    for d in review_decisions:
        d['review_pass'] = True
        s, e = d.get('start_ms'), d.get('end_ms')
        if s is not None and e is not None:
            duration = e - s
            confidence = d.get('confidence', 0) or 0
            desc = d.get('description', d.get('reason', 'Review pass suggestion'))
            # Apply high-confidence, short cuts (mispronunciations, clear stumbles)
            if confidence >= 88 and duration <= 5000:
                print(f"Review pass APPLYING: {s}ms-{e}ms ({confidence}%) — {desc[:80]}")
                applied_review += 1
            else:
                # Convert to advisory Note
                print(f"Review pass advisory: {s}ms-{e}ms ({confidence}%) — {desc[:80]}")
                d['original_start_ms'] = s
                d['original_end_ms'] = e
                del d['start_ms']
                del d['end_ms']
                d['type'] = 'Note'
                d['description'] = f"[REVIEW] {desc}"
                advisory_review += 1

    print(f"Review pass: {applied_review} cuts applied, {advisory_review} advisory notes")

    # Combine initial + review decisions
    all_decisions = edit_decisions + review_decisions

    return {
        "edit_decisions": all_decisions,
        "analysis_timestamp": datetime.now().isoformat(),
        "predetection": {
            "stumbles": stumbles,
            "meta_comments": meta_comments,
            "stutters": stutters,
            "fillers": fillers,
            "filler_pct": filler_pct,
        },
    }


# ============================================================================
# AUDIO EDITING — word-level edit model (no pydub, avoids OOM on large files)
# ============================================================================

# Strong opener phrases — 2-3 word sequences that reliably signal "on air"
_OPENER_PHRASES = [
    ('welcome', 'to'), ('welcome', 'back'), ('welcome', 'everyone'),
    ('welcome', 'ladies'), ('welcome', 'listeners'),
    ('hello', 'and'), ('hello', 'everyone'), ('hello', 'listeners'),
    ('hey', 'everyone'), ('hey', 'folks'), ('hey', 'guys'), ('hey', 'listeners'),
    ('hi', 'everyone'), ('hi', 'folks'), ('hi', 'guys'), ('hi', 'listeners'),
    ('hi', 'there'),
    ('good', 'morning'), ('good', 'afternoon'), ('good', 'evening'),
    ('thanks', 'for', 'joining'), ('thanks', 'for', 'tuning'),
    ('thank', 'you', 'for'),
    ('this', 'is', 'the'), ('this', 'is', 'episode'),
    ('today', 'we'), ("today's", 'episode'), ('today', 'on'),
    ('in', 'this', 'episode'), ('on', 'this', 'episode'),
    ("on", "today's", "episode"),
    ("i'm", 'your', 'host'), ('my', 'name', 'is'),
    ('episode', 'number'), ('episode', 'one'), ('episode', 'two'),
]
_PRECHAT_PHRASES = [
    ('are', 'we', 'recording'), ('is', 'it', 'recording'),
    ('hit', 'record'), ('start', 'recording'), ('stop', 'recording'),
    ('can', 'you', 'hear'), ('mic', 'check'), ('sound', 'check'),
    ('okay', 'recording'), ("we're", 'recording'), ('we', 'are', 'recording'),
    ("i'll", 'hit', 'record'), ('let', 'me', 'record'),
    ('are', 'you', 'all', 'set'), ('all', 'set'),
    ('ready', 'to', 'go'), ('shall', 'we', 'start'),
    ("let's", 'get', 'into'), ("let's", 'go'), ('here', 'we', 'go'),
    ("i'll", 'start', 'with'), ("let's", 'start'),
    ('are', 'you', 'ready'), ('you', 'ready'),
    ('up', 'and', 'running'),
]
_GREETING_WORDS = {'hi', 'hello', 'welcome'}


def _find_opener_ms(words):
    """Find the start_ms of the episode opener using phrase-based detection.
    Returns (opener_ms, has_prechat) or (None, False)."""
    tokens = []
    for w in words:
        ws = w.get('start', 0)
        if ws > 90000:
            break
        tok = w.get('text', '').lower().strip('.,!?;:\'"')
        tokens.append((tok, ws))
    if not tokens:
        return None, False

    # Detect pre-chat indicators
    has_prechat = False
    prechat_end_ms = 0
    for phrase in _PRECHAT_PHRASES:
        plen = len(phrase)
        for i in range(len(tokens) - plen + 1):
            if all(tokens[i + j][0] == phrase[j] for j in range(plen)):
                has_prechat = True
                prechat_end_ms = max(prechat_end_ms, tokens[i + plen - 1][1])
                print(f"Pre-chat indicator: '{' '.join(phrase)}' at {tokens[i][1]}ms")
                break

    # Find earliest opener phrase (after pre-chat if present)
    best_ms = None
    best_label = None
    for phrase in _OPENER_PHRASES:
        plen = len(phrase)
        for i in range(len(tokens) - plen + 1):
            if all(tokens[i + j][0] == phrase[j] for j in range(plen)):
                ms = tokens[i][1]
                if has_prechat and ms <= prechat_end_ms:
                    continue
                if best_ms is None or ms < best_ms:
                    best_ms = ms
                    best_label = f"phrase '{' '.join(phrase)}'"
                break

    # Check greeting + name (e.g. "Hi, L.J.")
    if has_prechat:
        skip = {'um', 'uh', 'and', 'the', 'a', 'so', 'but', 'or', 'yeah', 'yes',
                'no', 'okay', 'ok', 'right', 'well', 'like', 'just', 'i', 'we',
                'you', 'can', 'do', 'is', 'are', 'how', 'what', 'there', 'it'}
        for i in range(len(tokens) - 1):
            tok, ms = tokens[i]
            if ms <= prechat_end_ms:
                continue
            if tok in _GREETING_WORDS:
                next_tok = tokens[i + 1][0]
                if next_tok not in skip:
                    if best_ms is None or ms < best_ms:
                        best_ms = ms
                        best_label = f"greeting '{tok} {next_tok}'"
                    break

    if best_ms is not None:
        print(f"Opener detected: {best_label} at {best_ms}ms")
        return best_ms, has_prechat

    # Fallback: look for pause after last pre-chat word
    if has_prechat and prechat_end_ms > 0:
        for i in range(len(tokens) - 1):
            if tokens[i][1] < prechat_end_ms:
                continue
            if i + 1 < len(tokens):
                gap = tokens[i + 1][1] - tokens[i][1]
                if gap > 500:
                    print(f"Post-prechat pause {gap}ms, opener at {tokens[i+1][1]}ms")
                    return tokens[i + 1][1], has_prechat
        for i in range(len(tokens)):
            if tokens[i][1] > prechat_end_ms:
                return tokens[i][1], has_prechat

    return None, has_prechat


def _build_word_edit_map(words, cuts_ms, stumbles, meta_comments, stutters,
                         fillers, filler_pct, total_ms):
    """Phase 1: Mark each word as KEEP or CUT. No millisecond math.

    Returns words list with 'edit_action' ('keep'/'cut') and 'cut_reason' added.
    """
    # Start with every word marked KEEP
    for w in words:
        w['edit_action'] = 'keep'
        w['cut_reason'] = ''

    def _mark_range(start_ms, end_ms, reason):
        """Mark all words fully inside a time range as CUT."""
        count = 0
        for w in words:
            ws, we = w.get('start', 0), w.get('end', 0)
            if ws >= start_ms - 30 and we <= end_ms + 30 and we > ws:
                if w['edit_action'] != 'cut':
                    w['edit_action'] = 'cut'
                    w['cut_reason'] = reason
                    count += 1
        return count

    # 1. Pre-chat: mark all words before opener.
    # Two detection paths:
    #   a) _find_opener_ms detects pre-chat phrases AND an opener
    #   b) Claude included a pre-chat cut (start < 60s) — use that as confirmation
    # If EITHER path fires, mark all words before the opener as CUT.
    opener_ms, has_prechat = _find_opener_ms(words)

    # Check if Claude included a pre-chat cut (any cut starting near 0)
    claude_prechat_end = 0
    for s, e in cuts_ms:
        s, e = int(s), int(e)
        if s < 2000 and e > 2000:  # starts near beginning, extends past 2s
            claude_prechat_end = max(claude_prechat_end, e)

    # Use the best available information
    if opener_ms is None and claude_prechat_end > 0:
        # Claude found pre-chat but our phrase detection missed it.
        # Use Claude's cut end as the opener position.
        opener_ms = claude_prechat_end
        has_prechat = True
        print(f"Pre-chat: using Claude's cut end {claude_prechat_end}ms as opener (phrase detection missed it)")
    elif opener_ms is not None and claude_prechat_end > 0:
        # Both detected — use whichever gives the later cut point
        # (Claude may have found content our phrase detection missed)
        if claude_prechat_end > opener_ms:
            print(f"Pre-chat: Claude's cut extends past our opener ({claude_prechat_end}ms > {opener_ms}ms), using opener")
        has_prechat = True

    if opener_ms is not None and has_prechat:
        prechat_count = 0
        # Mark all words whose START is before the opener as CUT.
        # In word-level model we don't need a safety buffer — we just keep
        # any word that starts at or after the opener timestamp.
        for w in words:
            if w.get('start', 0) < opener_ms:
                w['edit_action'] = 'cut'
                w['cut_reason'] = 'prechat'
                prechat_count += 1
        if prechat_count:
            print(f"Word map: marked {prechat_count} pre-chat words as CUT (opener at {opener_ms}ms)")

    # 2. Pre-detected stumbles
    for stum in stumbles:
        s, e = int(stum['stumble_start_ms']), int(stum['clean_start_ms'])
        n = _mark_range(s, e, f"stumble:{stum.get('phrase', '')}")
        if n:
            print(f"Word map: stumble '{stum.get('phrase', '')}' — {n} words CUT ({s}-{e}ms)")

    # 3. Pre-detected stutters
    for stut in stutters:
        s, e = int(stut['start_ms']), int(stut['end_ms'])
        n = _mark_range(s, e, f"stutter:{stut.get('partial', '')}")
        if n:
            print(f"Word map: stutter '{stut.get('partial', '')}→{stut.get('full', '')}' — {n} words CUT")

    # 4. Pre-detected meta-commentary
    for meta in meta_comments:
        s, e = int(meta['start_ms']), int(meta['end_ms'])
        n = _mark_range(s, e, f"meta:{meta.get('text', '')[:30]}")
        if n:
            print(f"Word map: meta '{meta.get('text', '')[:40]}' — {n} words CUT")

    # 5. Disruptive fillers (up to filler_pct target)
    disruptive = [f for f in fillers if f.get('context') == 'disruptive']
    if disruptive:
        target = int(len(disruptive) * filler_pct / 100)
        already = sum(1 for f in disruptive
                      if any(w.get('start', 0) >= int(f['start_ms']) - 30 and
                             w.get('end', 0) <= int(f['end_ms']) + 30 and
                             w['edit_action'] == 'cut' for w in words))
        needed = target - already
        if needed > 0:
            uncovered = [f for f in disruptive
                         if not any(w.get('start', 0) >= int(f['start_ms']) - 30 and
                                    w.get('end', 0) <= int(f['end_ms']) + 30 and
                                    w['edit_action'] == 'cut' for w in words)]
            injected = 0
            for f in uncovered[:needed]:
                n = _mark_range(int(f['start_ms']), int(f['end_ms']),
                                f"filler:{f.get('text', '')}")
                if n:
                    injected += 1
            print(f"Word map: {already} fillers already cut, injected {injected} more "
                  f"(target {target}/{len(disruptive)}, {filler_pct}%)")

    # 6. Claude's content cuts — mark words fully inside each range as CUT.
    # Only words FULLY inside the range are cut. Partial overlap = KEEP.
    # Skip cuts > 10s mid-episode (likely Claude errors) unless they align with
    # pre-detected ranges.
    claude_cuts = 0
    for s, e in cuts_ms:
        s, e = int(s), int(e)
        if e <= s:
            continue
        duration = e - s
        is_start = s < 60000
        is_end = e > total_ms * 0.85
        if duration > 10000 and not is_start and not is_end:
            # Check if pre-detected ranges cover most of it
            known = sum(max(0, min(e, int(m.get('end_ms', 0))) -
                            max(s, int(m.get('start_ms', 0))))
                        for m in meta_comments)
            known += sum(max(0, min(e, int(st['clean_start_ms'])) -
                             max(s, int(st['stumble_start_ms'])))
                         for st in stumbles)
            if known < duration * 0.5:
                print(f"Word map: skipping oversized Claude cut {s}-{e}ms ({duration/1000:.1f}s)")
                continue
        n = _mark_range(s, e, 'claude')
        claude_cuts += n
    if claude_cuts:
        print(f"Word map: Claude marked {claude_cuts} additional words as CUT")

    # 7. Speaker boundary check: if a run of CUT words spans speakers, only cut
    # the dominant speaker's words.
    i = 0
    while i < len(words):
        if words[i]['edit_action'] != 'cut':
            i += 1
            continue
        # Find the run of CUT words
        run_start = i
        while i < len(words) and words[i]['edit_action'] == 'cut':
            i += 1
        run_end = i
        run_words = words[run_start:run_end]
        speakers = set(w.get('speaker', '?') for w in run_words)
        if len(speakers) > 1 and run_words[0].get('start', 0) > 60000:
            # Multiple speakers — only cut dominant speaker
            from collections import Counter
            counts = Counter(w.get('speaker', '?') for w in run_words)
            dominant = counts.most_common(1)[0][0]
            restored = 0
            for w in run_words:
                if w.get('speaker', '?') != dominant:
                    w['edit_action'] = 'keep'
                    w['cut_reason'] = ''
                    restored += 1
            if restored:
                print(f"Word map: speaker boundary — restored {restored} words from non-dominant speaker")

    # Summary
    cut_count = sum(1 for w in words if w['edit_action'] == 'cut')
    keep_count = sum(1 for w in words if w['edit_action'] == 'keep')
    print(f"Word edit map complete: {cut_count} CUT, {keep_count} KEEP out of {len(words)} words")

    return words


def _post_mark_verification(words):
    """Scan KEEP words for adjacent duplicates (same text, same speaker) that
    became neighbours after other words were cut. Mark the first as CUT.
    Also catches 2-word phrase duplicates."""
    fixes = 0
    keep_words = [(i, w) for i, w in enumerate(words) if w['edit_action'] == 'keep']

    # Single-word duplicates
    for k in range(len(keep_words) - 1):
        idx_a, wa = keep_words[k]
        idx_b, wb = keep_words[k + 1]
        if wa.get('speaker', '?') != wb.get('speaker', '?'):
            continue
        ta = wa.get('text', '').lower().strip('.,!?;:')
        tb = wb.get('text', '').lower().strip('.,!?;:')
        if ta == tb and len(ta) >= 2 and ta not in FILLER_WORDS:
            wa['edit_action'] = 'cut'
            wa['cut_reason'] = 'post-verify-dup'
            fixes += 1
            print(f"Post-verify: duplicate '{ta}' at {wa.get('start', 0)}ms — marking first as CUT")

    # 2-word phrase duplicates
    keep_words = [(i, w) for i, w in enumerate(words) if w['edit_action'] == 'keep']
    for k in range(len(keep_words) - 3):
        _, wa = keep_words[k]
        _, wb = keep_words[k + 1]
        _, wc = keep_words[k + 2]
        _, wd = keep_words[k + 3]
        if wa.get('speaker', '?') != wc.get('speaker', '?'):
            continue
        p1 = (wa.get('text', '').lower().strip('.,!?;:') + ' ' +
              wb.get('text', '').lower().strip('.,!?;:'))
        p2 = (wc.get('text', '').lower().strip('.,!?;:') + ' ' +
              wd.get('text', '').lower().strip('.,!?;:'))
        if p1 == p2 and len(p1) >= 5:
            wa['edit_action'] = 'cut'
            wa['cut_reason'] = 'post-verify-dup'
            wb['edit_action'] = 'cut'
            wb['cut_reason'] = 'post-verify-dup'
            fixes += 1
            print(f"Post-verify: duplicate phrase '{p1}' at {wa.get('start', 0)}ms — marking first pair as CUT")

    if fixes:
        print(f"Post-verify: fixed {fixes} duplicate(s)")
    else:
        print("Post-verify: no new duplicates found")
    return fixes


def _compute_audio_segments(words, total_ms):
    """Phase 2: From the KEEP/CUT word marks, compute audio keep-segments.

    Key principle: each segment boundary is computed FROM word timestamps.
    The gap between the last KEEP word and the first CUT word is split,
    preserving natural room tone and preventing clipping.

    Returns list of (start_ms, end_ms) keep-segments ready for ffmpeg.
    """
    TAIL_MS = 15   # preserve after last KEEP word before a cut
    HEAD_MS = 15   # preserve before first KEEP word after a cut
    MIN_GAP_MS = 60  # minimum gap to leave in the edit (natural pacing)

    # Build runs of consecutive KEEP words (a "segment")
    segments = []  # list of {'words': [...], 'first_idx': int, 'last_idx': int}
    current_seg = None
    for i, w in enumerate(words):
        if w['edit_action'] == 'keep':
            if current_seg is None:
                current_seg = {'words': [w], 'first_idx': i, 'last_idx': i}
            else:
                current_seg['words'].append(w)
                current_seg['last_idx'] = i
        else:
            if current_seg is not None:
                segments.append(current_seg)
                current_seg = None
    if current_seg is not None:
        segments.append(current_seg)

    if not segments:
        return [(0, min(1000, total_ms))]

    # Convert word-segments to audio time ranges
    keeps = []
    for seg_idx, seg in enumerate(segments):
        first_word = seg['words'][0]
        last_word = seg['words'][-1]

        # Segment audio start
        if seg_idx == 0 and seg['first_idx'] == 0:
            # First segment AND first word is KEEP: start from beginning of audio
            seg_start = 0
        elif seg_idx == 0:
            # First segment but there are CUT words before it (e.g. pre-chat).
            # Start just before the first KEEP word — don't include pre-chat audio.
            word_start = first_word.get('start', 0)
            seg_start = max(0, word_start - HEAD_MS)
        else:
            # Find the last CUT word before this segment
            prev_cut_end = 0
            for j in range(seg['first_idx'] - 1, -1, -1):
                if words[j]['edit_action'] == 'cut':
                    prev_cut_end = words[j].get('end', 0)
                    break
            # Start: preserve HEAD_MS before first word, but don't go before
            # the previous cut word's end + a small gap for room tone
            word_start = first_word.get('start', 0)
            # Use the midpoint of the gap between cut end and keep start,
            # biased toward preserving the keep word's onset
            gap = word_start - prev_cut_end
            if gap > MIN_GAP_MS:
                # Leave MIN_GAP_MS/2 after the cut, take the rest
                seg_start = max(prev_cut_end + MIN_GAP_MS // 2,
                                word_start - HEAD_MS)
            else:
                seg_start = max(prev_cut_end, word_start - HEAD_MS)

        # Segment audio end
        if seg_idx == len(segments) - 1:
            # Last segment: go to end of audio
            seg_end = total_ms
        else:
            # Find the first CUT word after this segment
            next_cut_start = total_ms
            for j in range(seg['last_idx'] + 1, len(words)):
                if words[j]['edit_action'] == 'cut':
                    next_cut_start = words[j].get('start', total_ms)
                    break
            # End: preserve TAIL_MS after last word, but don't go past
            # the next cut word's start
            word_end = last_word.get('end', 0)
            gap = next_cut_start - word_end
            if gap > MIN_GAP_MS:
                seg_end = min(next_cut_start - MIN_GAP_MS // 2,
                              word_end + TAIL_MS)
            else:
                seg_end = min(next_cut_start, word_end + TAIL_MS)

        if seg_end > seg_start:
            keeps.append((int(seg_start), int(seg_end)))

    # Handle gap-only cuts (non-verbal gaps between words that are both KEEP).
    # These are audio regions between KEEP words where coughs/noises live.
    # The _find_nonverbal_gaps detection already identified them. We need to
    # split any keep-segment that spans such a gap.
    # (This is handled naturally because non-verbal gap trims are converted to
    # word-level "cut" marks on the gap region. Since there are no words in the
    # gap, the gap audio stays in the segment. We handle this explicitly here.)
    # For now, the gap trims from _find_nonverbal_gaps are handled by marking
    # a virtual gap region. Since there are no words there, the segment
    # boundary computation already trims into the gap via TAIL_MS/HEAD_MS.

    # Filter tiny segments
    keeps = [(s, e) for s, e in keeps if e - s >= 100]
    if not keeps:
        keeps = [(0, min(1000, total_ms))]

    return keeps


# ============================================================================
# PHASE 3 + 4: ffmpeg export and iterative join refinement
# ============================================================================

def _ffmpeg_concat_segments(audio_path, keeps, fade_ms=8, per_segment_fade=None):
    """Phase 3: trim each segment, apply micro-fades, concatenate.

    Args:
        audio_path: source audio file
        keeps: list of (start_ms, end_ms) tuples
        fade_ms: default fade duration in ms (8ms = inaudible click prevention)
        per_segment_fade: optional dict mapping segment index to custom fade_ms

    Returns: path to exported WAV
    """
    if per_segment_fade is None:
        per_segment_fade = {}

    filter_parts = []
    for i, (start, end) in enumerate(keeps):
        seg_fade = per_segment_fade.get(i, fade_ms)
        fade_s = seg_fade / 1000
        start_s = start / 1000
        end_s = end / 1000
        seg_dur = end_s - start_s
        chain = [f"atrim={start_s}:{end_s}", "asetpts=N/SR/TB"]
        if i > 0 and seg_dur > fade_s * 3:
            chain.append(f"afade=t=in:d={fade_s}")
        if i < len(keeps) - 1 and seg_dur > fade_s * 3:
            fade_start = max(0, seg_dur - fade_s)
            chain.append(f"afade=t=out:st={fade_start:.4f}:d={fade_s}")
        filter_parts.append(f"[0:a]{','.join(chain)}[s{i}]")

    if len(keeps) == 1:
        full_filter = ";".join(filter_parts) + ";[s0]acopy[out]"
    else:
        full_filter = ";".join(filter_parts)
        labels = "".join(f"[s{i}]" for i in range(len(keeps)))
        full_filter += f";{labels}concat=n={len(keeps)}:v=0:a=1[out]"

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

    remaining_ms = sum(e - s for s, e in keeps)
    print(f"Exported: {wav_path} ({remaining_ms}ms, {os.path.getsize(wav_path) // 1024}KB)")
    return wav_path


def _analyze_join_points(wav_path, keeps):
    """Analyze each join point in edited WAV for acoustic issues.

    Measures RMS level in 150ms windows before/after each join to detect
    harsh transitions. Returns list of issues with join index and severity.
    """
    # Calculate output-timeline positions of each join
    output_pos_ms = 0
    joins = []  # (join_idx, output_ms)
    for i in range(len(keeps)):
        if i > 0:
            joins.append((i, output_pos_ms))
        output_pos_ms += keeps[i][1] - keeps[i][0]

    if not joins:
        return []

    # Sample up to 30 join points (evenly spaced if more)
    if len(joins) > 30:
        step = len(joins) / 30
        sampled = [joins[int(i * step)] for i in range(30)]
    else:
        sampled = joins

    issues = []
    for join_idx, join_ms in sampled:
        try:
            before_s = max(0, (join_ms - 150) / 1000)
            before_e = join_ms / 1000
            after_s = join_ms / 1000
            after_e = (join_ms + 150) / 1000

            rms_vals = []
            for ss, se in [(before_s, before_e), (after_s, after_e)]:
                r = subprocess.run(
                    ['ffmpeg', '-i', wav_path, '-ss', f'{ss:.3f}',
                     '-to', f'{se:.3f}', '-af', 'volumedetect',
                     '-f', 'null', '-'],
                    capture_output=True, text=True, timeout=10,
                )
                m = re.search(r'mean_volume:\s*(-?[\d.]+)', r.stderr)
                rms_vals.append(float(m.group(1)) if m else -60.0)

            if len(rms_vals) == 2:
                rms_diff = abs(rms_vals[0] - rms_vals[1])
                if rms_diff > 6.0:
                    issues.append({
                        'join_idx': join_idx,
                        'output_ms': join_ms,
                        'issue': 'rms_jump',
                        'rms_diff_db': rms_diff,
                        'rms_before': rms_vals[0],
                        'rms_after': rms_vals[1],
                    })
        except Exception:
            pass  # don't fail the edit for analysis errors

    return issues


def _verify_edit_transcription(wav_path, expected_words, keeps):
    """Re-transcribe edited audio and compare against expected words.

    Sends the edited WAV to AssemblyAI, gets a fresh transcript, and checks
    for missing or garbled words near cut join points. Returns list of issues.
    """
    # Build output-timeline join positions
    output_joins_ms = set()
    pos = 0
    for i, (s, e) in enumerate(keeps):
        if i > 0:
            output_joins_ms.add(pos)
        pos += e - s

    # Upload and transcribe the edited audio
    try:
        with open(wav_path, 'rb') as f:
            audio_data = f.read()
        upload_url = stream_bytes_to_assemblyai(audio_data)
        # Start transcription with simpler config (no disfluencies needed)
        resp = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json={
                "audio_url": upload_url,
                "speech_models": ["universal-2"],
                "punctuate": True,
                "format_text": True,
            },
            headers={
                "authorization": ASSEMBLYAI_API_KEY,
                "content-type": "application/json",
            },
        )
        if resp.status_code != 200:
            print(f"Re-transcription submit failed: {resp.status_code}")
            return []
        tx_id = resp.json().get('id')
        if not tx_id:
            return []

        # Poll for completion (max 120s)
        for _ in range(40):
            time.sleep(3)
            tx_data = get_transcription(tx_id)
            status = tx_data.get('status')
            if status == 'completed':
                break
            if status == 'error':
                print(f"Re-transcription failed: {tx_data.get('error')}")
                return []
        else:
            print("Re-transcription timed out after 120s")
            return []

        new_words = tx_data.get('words', [])
        if not new_words:
            return []

    except Exception as e:
        print(f"Re-transcription error: {e}")
        return []

    # Align re-transcribed words to expected words and find issues near joins
    from difflib import SequenceMatcher

    # Build expected word list with output-timeline timestamps
    expected = []
    seg_offset = 0
    seg_idx = 0
    for w in expected_words:
        ws = w.get('start', 0)
        # Find which segment this word falls in
        while seg_idx < len(keeps) and ws >= keeps[seg_idx][1]:
            seg_offset += keeps[seg_idx][1] - keeps[seg_idx][0]
            seg_idx += 1
        if seg_idx < len(keeps):
            output_ms = seg_offset + (ws - keeps[seg_idx][0])
        else:
            output_ms = seg_offset
        expected.append({
            'text': w.get('text', '').lower().strip('.,!?;:\'"'),
            'output_ms': output_ms,
        })

    # Walk both lists with sliding window alignment
    issues = []
    new_idx = 0
    for exp in expected:
        # Is this word near a join point? (within 300ms)
        near_join = any(abs(exp['output_ms'] - j) < 300 for j in output_joins_ms)
        if not near_join:
            continue

        # Find best match in new transcript within a window
        best_ratio = 0
        best_conf = 1.0
        best_text = ''
        search_ms = exp['output_ms']
        for ni in range(max(0, new_idx - 3), min(len(new_words), new_idx + 6)):
            nw = new_words[ni]
            nw_text = nw.get('text', '').lower().strip('.,!?;:\'"')
            ratio = SequenceMatcher(None, exp['text'], nw_text).ratio()
            # Also check timing proximity (within 500ms)
            nw_start = nw.get('start', 0)
            if abs(nw_start - search_ms) < 500 and ratio > best_ratio:
                best_ratio = ratio
                best_conf = nw.get('confidence', 1.0)
                best_text = nw_text

        # Find the join this word is closest to
        closest_join_idx = 0
        closest_dist = float('inf')
        pos = 0
        for ki in range(len(keeps)):
            if ki > 0:
                d = abs(exp['output_ms'] - pos)
                if d < closest_dist:
                    closest_dist = d
                    closest_join_idx = ki
            pos += keeps[ki][1] - keeps[ki][0]

        if best_ratio < 0.5:
            issues.append({
                'join_idx': closest_join_idx,
                'output_ms': exp['output_ms'],
                'issue': 'missing_word',
                'expected_word': exp['text'],
                'got_word': best_text,
                'confidence': best_conf,
            })
        elif best_conf < 0.6:
            issues.append({
                'join_idx': closest_join_idx,
                'output_ms': exp['output_ms'],
                'issue': 'low_confidence',
                'expected_word': exp['text'],
                'got_word': best_text,
                'confidence': best_conf,
            })

        # Advance new_idx roughly
        if best_ratio > 0.5:
            for ni in range(max(0, new_idx - 3), min(len(new_words), new_idx + 6)):
                nw_text = new_words[ni].get('text', '').lower().strip('.,!?;:\'"')
                if SequenceMatcher(None, exp['text'], nw_text).ratio() == best_ratio:
                    new_idx = ni + 1
                    break

    return issues


def _compute_boundary_fixes(acoustic_issues, transcript_issues, keeps):
    """Given detected issues, compute boundary adjustments.

    Only adjusts the physical audio boundaries — never changes word-level
    KEEP/CUT decisions. Fixes are conservative: widen fades first, shift
    boundaries only if needed.

    Returns:
        adjusted_keeps: new list of (start_ms, end_ms) tuples
        fixes_applied: list of description strings
        per_segment_fade: dict mapping segment index to fade_ms override
    """
    adjusted = list(keeps)
    per_segment_fade = {}
    fixes = []

    # Collect all problematic join indices
    issues_by_join = {}
    for issue in acoustic_issues + transcript_issues:
        idx = issue['join_idx']
        if idx not in issues_by_join:
            issues_by_join[idx] = []
        issues_by_join[idx].append(issue)

    for join_idx, join_issues in sorted(issues_by_join.items()):
        issue_types = [i['issue'] for i in join_issues]
        has_rms = 'rms_jump' in issue_types
        has_missing = 'missing_word' in issue_types or 'low_confidence' in issue_types

        prev_idx = join_idx - 1
        if prev_idx < 0 or join_idx >= len(adjusted):
            continue

        prev_start, prev_end = adjusted[prev_idx]
        curr_start, curr_end = adjusted[join_idx]

        # Step 1: Always widen fade for acoustic issues
        if has_rms:
            per_segment_fade[prev_idx] = max(per_segment_fade.get(prev_idx, 8), 20)
            per_segment_fade[join_idx] = max(per_segment_fade.get(join_idx, 8), 20)
            rms_issue = next(i for i in join_issues if i['issue'] == 'rms_jump')
            fixes.append(
                f"Join {join_idx}: widen fade 8→20ms "
                f"(RMS jump {rms_issue['rms_diff_db']:.1f}dB)")

        # Step 2: Shift boundaries for missing/garbled words
        if has_missing:
            for issue in join_issues:
                if issue['issue'] not in ('missing_word', 'low_confidence'):
                    continue
                word = issue.get('expected_word', '?')
                # Determine which side the word is on: end of prev or start of curr
                # If output_ms is before the join, word is at end of prev segment
                output_ms = issue.get('output_ms', 0)
                pos = sum(adjusted[k][1] - adjusted[k][0] for k in range(join_idx))

                SHIFT = 30  # ms
                if output_ms < pos:
                    # Word at end of previous segment — extend it
                    new_end = min(prev_end + SHIFT, curr_start - 30)
                    if new_end > prev_end and new_end - prev_start >= 100:
                        adjusted[prev_idx] = (prev_start, new_end)
                        fixes.append(
                            f"Join {join_idx}: extend prev segment +{SHIFT}ms "
                            f"('{word}' at end)")
                else:
                    # Word at start of current segment — start earlier
                    new_start = max(curr_start - SHIFT, prev_end + 30)
                    if new_start < curr_start and curr_end - new_start >= 100:
                        adjusted[join_idx] = (new_start, curr_end)
                        fixes.append(
                            f"Join {join_idx}: shift start -{SHIFT}ms "
                            f"('{word}' at start)")

        # Step 3: For RMS jumps that are very harsh (>10dB), also shift
        if has_rms and not has_missing:
            rms_issue = next(i for i in join_issues if i['issue'] == 'rms_jump')
            if rms_issue['rms_diff_db'] > 10.0:
                SHIFT = 30
                # Trim from the louder side
                if rms_issue['rms_before'] > rms_issue['rms_after']:
                    # Loud before, quiet after — trim end of prev
                    new_end = max(prev_start + 100, prev_end - SHIFT)
                    if new_end < prev_end:
                        adjusted[prev_idx] = (prev_start, new_end)
                        fixes.append(
                            f"Join {join_idx}: trim prev segment -{SHIFT}ms "
                            f"(RMS {rms_issue['rms_diff_db']:.1f}dB)")
                else:
                    # Quiet before, loud after — trim start of curr
                    new_start = min(curr_end - 100, curr_start + SHIFT)
                    if new_start > curr_start:
                        adjusted[join_idx] = (new_start, curr_end)
                        fixes.append(
                            f"Join {join_idx}: trim curr start +{SHIFT}ms "
                            f"(RMS {rms_issue['rms_diff_db']:.1f}dB)")

    return adjusted, fixes, per_segment_fade


def _iterate_join_refinement(audio_path, wav_path, keeps, words):
    """Phase 4: Verify join quality and iteratively refine boundaries.

    After the initial ffmpeg export, analyzes each join point for acoustic
    issues (RMS jumps, clicks) and optionally re-transcribes to check for
    clipped words. Applies fixes (wider fades, shifted boundaries) and
    re-exports. Max 3 iterations — each fixes fewer issues than the last.

    Word-level KEEP/CUT decisions are never changed. Only the physical
    audio boundaries (keeps list) are adjusted.
    """
    MAX_ITERATIONS = 3
    verify_tx = os.environ.get('VERIFY_EDIT_TRANSCRIPTION') == '1'
    expected_keep_words = [w for w in words if w.get('edit_action') == 'keep']

    for iteration in range(MAX_ITERATIONS):
        # Acoustic analysis at join points
        acoustic_issues = _analyze_join_points(wav_path, keeps)

        # Optional re-transcription verification (first iteration only — expensive)
        transcript_issues = []
        if verify_tx and iteration == 0:
            print("Re-transcribing edited audio for verification...")
            transcript_issues = _verify_edit_transcription(
                wav_path, expected_keep_words, keeps)
            if transcript_issues:
                print(f"Re-transcription found {len(transcript_issues)} issues near joins")

        all_issues = acoustic_issues + transcript_issues
        if not all_issues:
            if iteration == 0:
                print(f"Join refinement: all {len(keeps) - 1} joins pass on first check")
            else:
                print(f"Join refinement: clean after {iteration + 1} iterations")
            break

        print(f"Join refinement pass {iteration + 1}: "
              f"{len(acoustic_issues)} acoustic + {len(transcript_issues)} transcript issues")

        # Compute fixes
        keeps, fixes, per_segment_fade = _compute_boundary_fixes(
            acoustic_issues, transcript_issues, keeps)

        if not fixes:
            print(f"Join refinement: no fixable issues, stopping")
            break

        for desc in fixes:
            print(f"  {desc}")

        # Re-export with adjusted boundaries
        wav_path = _ffmpeg_concat_segments(
            audio_path, keeps, per_segment_fade=per_segment_fade)
    else:
        print(f"Join refinement: reached max iterations ({MAX_ITERATIONS})")

    return keeps, wav_path


def apply_audio_edits(audio_path, cuts_ms, words=None, transcript_id=None):
    """
    Remove segments from audio using word-level edit model + ffmpeg.

    Architecture: instead of mutating millisecond ranges through 14 competing
    steps, we mark each word as KEEP or CUT, then compute audio boundaries
    FROM word timestamps. This eliminates the class of bugs where modifications
    interact and swallow content.

    Phase 1: _build_word_edit_map — mark words KEEP/CUT
    Phase 2: _compute_audio_segments — compute audio boundaries from word marks
    Phase 3: ffmpeg filter_complex — trim + micro-fade + concatenate
    """
    if words:
        words = _normalize_words(words)

    # Get audio duration
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', audio_path],
        capture_output=True, text=True, check=True, timeout=30,
    )
    total_ms = int(float(probe.stdout.strip()) * 1000)

    # ── Retrieve cached pre-detection results or recompute ──
    cached_predetection = None
    if transcript_id:
        with _predetection_cache_lock:
            cached_predetection = _predetection_cache.get(transcript_id)
    if cached_predetection:
        print(f"Using cached pre-detection for {transcript_id}")
        stumbles = cached_predetection['stumbles']
        meta_comments = cached_predetection['meta_comments']
        stutters = cached_predetection['stutters']
        fillers = cached_predetection['fillers']
        filler_pct = cached_predetection.get('filler_pct', 60)
    elif words:
        print("No cached pre-detection — recomputing")
        stumbles = _find_stumbles(words)
        meta_comments = _find_meta_commentary(words)
        stutters = _find_stutters(words)
        fillers = _find_fillers(words)
        filler_pct = 60
    else:
        stumbles, meta_comments, stutters, fillers = [], [], [], []
        filler_pct = 60

    # ── PHASE 1: Build word-level edit map ──
    if words:
        words = _build_word_edit_map(
            words, cuts_ms, stumbles, meta_comments, stutters,
            fillers, filler_pct, total_ms,
        )
        _post_mark_verification(words)

        # ── Content diff: log what the listener hears at each cut boundary ──
        cut_runs = []
        in_cut = False
        run_start = 0
        for i, w in enumerate(words):
            if w['edit_action'] == 'cut' and not in_cut:
                in_cut = True
                run_start = i
            elif w['edit_action'] == 'keep' and in_cut:
                in_cut = False
                cut_runs.append((run_start, i - 1))
        if in_cut:
            cut_runs.append((run_start, len(words) - 1))

        print(f"--- Content diff ({len(cut_runs)} cuts) ---")
        for first_idx, last_idx in cut_runs:
            # 3 words before
            before = []
            for j in range(first_idx - 1, -1, -1):
                if words[j]['edit_action'] == 'keep':
                    before.insert(0, words[j].get('text', ''))
                    if len(before) >= 3:
                        break
            before_text = ' '.join(before)
            # Words removed
            removed_text = ' '.join(words[k].get('text', '') for k in range(first_idx, last_idx + 1))
            # 3 words after
            after = []
            for j in range(last_idx + 1, len(words)):
                if words[j]['edit_action'] == 'keep':
                    after.append(words[j].get('text', ''))
                    if len(after) >= 3:
                        break
            after_text = ' '.join(after)
            start_ms = words[first_idx].get('start', 0)
            end_ms = words[last_idx].get('end', 0)
            reason = words[first_idx].get('cut_reason', '')
            print(f"  [{reason}] {start_ms}-{end_ms}ms: ...{before_text} [CUT {removed_text}] {after_text}...")
        print("--- End content diff ---")

        # ── PHASE 1b: Claude validates the cut decisions ──
        # Build post-edit transcript from KEEP words and ask Claude to verify
        # it reads naturally with no context lost or stumbles remaining.
        try:
            validation_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

            # Build original transcript (first 8000 chars)
            original_lines = []
            current_speaker = None
            current_words = []
            for w in words:
                sp = w.get('speaker', '?')
                if sp != current_speaker:
                    if current_words:
                        original_lines.append(f"Speaker {current_speaker}: {' '.join(current_words)}")
                    current_speaker = sp
                    current_words = []
                current_words.append(w.get('text', ''))
            if current_words:
                original_lines.append(f"Speaker {current_speaker}: {' '.join(current_words)}")
            original_transcript = '\n'.join(original_lines)[:8000]

            # Build post-edit transcript from KEEP words with [CUT] markers
            post_lines = []
            current_speaker = None
            current_words = []
            in_cut = False
            for w in words:
                sp = w.get('speaker', '?')
                if w['edit_action'] == 'cut':
                    if not in_cut:
                        current_words.append('[CUT]')
                        in_cut = True
                    continue
                in_cut = False
                if sp != current_speaker:
                    if current_words:
                        post_lines.append(f"Speaker {current_speaker}: {' '.join(current_words)}")
                    current_speaker = sp
                    current_words = []
                current_words.append(f"{w.get('text', '')}({w.get('start', 0)})")
            if current_words:
                post_lines.append(f"Speaker {current_speaker}: {' '.join(current_words)}")
            post_transcript = '\n'.join(post_lines)[:8000]

            validation_prompt = f"""You are a podcast editor doing a FINAL QUALITY CHECK on edit decisions BEFORE they are applied to audio. Compare the original transcript with the post-edit version.

ORIGINAL TRANSCRIPT:
{original_transcript}

POST-EDIT TRANSCRIPT (what the listener will hear — [CUT] shows where content was removed, timestamps in parentheses):
{post_transcript}

Check for these specific problems:

1. CONTEXT LOSS: Did any cut remove words that change the meaning? For example, if "TA leaders" became just "leaders" because "TA" was cut. Look at each [CUT] marker — do the words before and after it still make sense together?

2. REMAINING STUMBLES: Are there any repeated phrases or false starts in the KEEP words that should have been cut? Look for back-to-back identical phrases.

3. BROKEN SENTENCES: Does any [CUT] create a grammatically broken or nonsensical sentence?

4. OVER-CUTTING: Was any substantive content removed that wasn't a filler, stumble, or disfluency?

Return your response as a JSON object with this structure:
{{"issues": [
  {{"type": "context_loss|remaining_stumble|broken_sentence|over_cut",
    "timestamp_ms": <start ms of the affected word>,
    "description": "brief description",
    "fix": "restore_word|cut_word",
    "word_text": "the affected word",
    "word_start_ms": <start ms to match>}}
], "verdict": "pass|has_issues"}}

If everything looks clean, return: {{"issues": [], "verdict": "pass"}}

IMPORTANT: Only flag CLEAR problems. Natural speech patterns, minor awkwardness, and stylistic choices are fine. Be conservative — false positives waste time."""

            print("Running cut decision validation with Claude...")
            val_response = validation_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": validation_prompt}],
            )
            val_text = val_response.content[0].text.strip()

            # Parse JSON from response (handle markdown code blocks)
            import json as _json
            if val_text.startswith('```'):
                val_text = val_text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            try:
                val_result = _json.loads(val_text)
            except _json.JSONDecodeError:
                # Try to find JSON in the response
                json_start = val_text.find('{')
                json_end = val_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    val_result = _json.loads(val_text[json_start:json_end])
                else:
                    val_result = {"issues": [], "verdict": "parse_error"}

            verdict = val_result.get('verdict', 'unknown')
            issues = val_result.get('issues', [])

            if verdict == 'pass' or not issues:
                print("Cut validation: PASS — all decisions look clean")
            else:
                print(f"Cut validation: {len(issues)} issue(s) found")
                for issue in issues:
                    itype = issue.get('type', '?')
                    desc = issue.get('description', '')
                    fix = issue.get('fix', '')
                    word_text = issue.get('word_text', '')
                    word_ms = issue.get('word_start_ms', 0)
                    print(f"  [{itype}] {desc} (word='{word_text}' at {word_ms}ms, fix={fix})")

                    # Apply fix
                    if fix == 'restore_word' and word_text:
                        # Find the word and flip it to KEEP
                        for w in words:
                            if (w.get('text', '').lower().strip('.,!?;:') ==
                                    word_text.lower().strip('.,!?;:') and
                                    abs(w.get('start', 0) - word_ms) < 200 and
                                    w['edit_action'] == 'cut'):
                                w['edit_action'] = 'keep'
                                w['cut_reason'] = ''
                                print(f"    RESTORED: '{word_text}' at {w.get('start', 0)}ms")
                                break
                    elif fix == 'cut_word' and word_text:
                        # Find the word and flip it to CUT
                        for w in words:
                            if (w.get('text', '').lower().strip('.,!?;:') ==
                                    word_text.lower().strip('.,!?;:') and
                                    abs(w.get('start', 0) - word_ms) < 200 and
                                    w['edit_action'] == 'keep'):
                                w['edit_action'] = 'cut'
                                w['cut_reason'] = f'validation:{itype}'
                                print(f"    CUT: '{word_text}' at {w.get('start', 0)}ms")
                                break
        except Exception as e:
            print(f"Cut validation skipped (non-fatal): {e}")

        # ── PHASE 2: Compute audio segments from word marks ──
        keeps = _compute_audio_segments(words, total_ms)

        # Apply nonverbal gap trims (coughs/throat clears between KEEP words).
        # These are gap-only cuts — no words to mark. Split any segment that
        # spans a detected gap.
        nonverbal_gaps = _find_nonverbal_gaps(words)
        if nonverbal_gaps:
            gap_trims = [(int(g['start_ms']), int(g['end_ms'])) for g in nonverbal_gaps]
            new_keeps = []
            for seg_start, seg_end in keeps:
                # Check if any gap trim falls inside this segment
                splits = []
                for gs, ge in gap_trims:
                    if gs > seg_start and ge < seg_end:
                        splits.append((gs, ge))
                if not splits:
                    new_keeps.append((seg_start, seg_end))
                else:
                    # Split the segment around the gap trims
                    splits.sort()
                    pos = seg_start
                    for gs, ge in splits:
                        if gs > pos:
                            new_keeps.append((pos, gs))
                        pos = ge
                    if pos < seg_end:
                        new_keeps.append((pos, seg_end))
            keeps = [(s, e) for s, e in new_keeps if e - s >= 100]
            if not keeps:
                keeps = [(0, min(1000, total_ms))]
            if len(keeps) != len(new_keeps):
                print(f"Gap trims: split segments for {len(nonverbal_gaps)} nonverbal gaps")
    else:
        # Fallback: no words available — use raw millisecond cuts (legacy path)
        adjusted = []
        for s, e in cuts_ms:
            s, e = int(s), int(e)
            if e > s:
                adjusted.append((max(0, s), min(total_ms, e)))
        adjusted.sort()
        merged = []
        for s, e in adjusted:
            if merged and s <= merged[-1][1] + 150:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        keeps = []
        pos = 0
        for s, e in merged:
            if s > pos:
                keeps.append((pos, s))
            pos = e
        if pos < total_ms:
            keeps.append((pos, total_ms))
        if not keeps:
            keeps = [(0, min(1000, total_ms))]
        keeps = [(s, e) for s, e in keeps if e - s >= 100]
        if not keeps:
            keeps = [(0, min(1000, total_ms))]

    removed_ms = total_ms - sum(e - s for s, e in keeps)
    remaining_ms = total_ms - removed_ms
    print(f"Segments: {len(keeps)} keep-segments, removed {removed_ms}ms, {remaining_ms}ms remaining")

    # ── PHASE 3: ffmpeg export + iteration loop ──
    wav_path = _ffmpeg_concat_segments(audio_path, keeps)

    # ── PHASE 4: Iteration loop — verify and refine join points ──
    if len(keeps) > 1 and words:
        keeps, wav_path = _iterate_join_refinement(audio_path, wav_path, keeps, words)

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




def _run_multitrack_job(job_id, track_paths, labels):
    """Background thread: combine raw tracks, upload to AssemblyAI, preserve track paths for cleanup."""
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

        # Preserve raw track paths for cleanup after editing
        with _multitrack_meta_lock:
            _multitrack_meta[transcript_id] = {
                'track_paths': list(track_paths),
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
    try:
        signed_id = upload_data['signed_id']
        signed_url = upload_data['direct_upload']['url']
    except (KeyError, TypeError) as e:
        raise Exception(f'Adobe upload response missing expected fields: {e} — got: {str(upload_data)[:200]}')
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
    speech_pct = max(0, min(100, int(mix.get('speech', 90))))
    bg_pct = max(0, min(100, int(mix.get('background', 10))))
    music_pct = max(0, min(100, int(mix.get('music', 10))))
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

def _analyze_for_enhance(wav_path):
    """Quick ffmpeg analysis of cut audio for enhance mix recommendation.
    Returns noise_floor_db, integrated_lufs, duration_s.
    Never raises — returns safe defaults on any failure."""
    result = {'noise_floor_db': -35.0, 'integrated_lufs': -24.0, 'duration_s': 0.0}

    try:
        # Duration via ffprobe
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', wav_path],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            result['duration_s'] = round(float(probe.stdout.strip()), 1)
    except Exception as e:
        print(f"Pre-enhance analysis: ffprobe failed: {e}")

    try:
        # ebur128 for integrated loudness
        ebur = subprocess.run(
            ['ffmpeg', '-hide_banner', '-i', wav_path,
             '-af', 'ebur128=peak=true', '-f', 'null', '-'],
            capture_output=True, text=True, timeout=120,
        )
        if ebur.returncode == 0:
            summary_start = ebur.stderr.rfind('Summary:')
            if summary_start >= 0:
                summary = ebur.stderr[summary_start:]
                m = re.search(r'I:\s*([-\d.]+)\s*LUFS', summary)
                if m:
                    result['integrated_lufs'] = round(float(m.group(1)), 1)
    except Exception as e:
        print(f"Pre-enhance analysis: ebur128 failed: {e}")

    try:
        # astats for noise floor (RMS trough = quietest segments)
        astats = subprocess.run(
            ['ffmpeg', '-hide_banner', '-i', wav_path,
             '-af', 'astats', '-f', 'null', '-'],
            capture_output=True, text=True, timeout=120,
        )
        if astats.returncode == 0:
            for line in astats.stderr.split('\n'):
                m = re.search(r'RMS trough dB:\s*([-\d.]+)', line)
                if m:
                    result['noise_floor_db'] = round(float(m.group(1)), 1)
                    break
    except Exception as e:
        print(f"Pre-enhance analysis: astats failed: {e}")

    print(f"Pre-enhance analysis: noise={result['noise_floor_db']}dB, "
          f"lufs={result['integrated_lufs']}, dur={result['duration_s']}s")
    return result


def _run_edit_job(job_id, audio_path, cuts_ms, transcript_id=None,
                  intro_path=None, outro_path=None):
    """
    Background thread: apply cuts, enhance, optionally merge intro/outro, finalize.
    Pipeline: cuts → Adobe enhance → merge (if intro/outro) → mastering → review → completed
    Intro/outro are already produced audio — they skip Adobe Enhance.
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
        wav_path = apply_audio_edits(audio_path, cuts_ms, words=words, transcript_id=transcript_id)

        # Analyze cut audio and pause for user approval of enhance mix
        current_step = 'awaiting_enhance'
        analysis = _analyze_for_enhance(wav_path)
        nf = analysis['noise_floor_db']
        if nf < -50:
            recommended_mix = {'speech': 100, 'background': 0, 'music': 0}
        elif nf > -35:
            recommended_mix = {'speech': 80, 'background': 15, 'music': 10}
        else:
            recommended_mix = {'speech': 90, 'background': 10, 'music': 10}

        approve_event = threading.Event()
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'awaiting_enhance'
            _edit_jobs[job_id]['analysis'] = analysis
            _edit_jobs[job_id]['recommended_mix'] = recommended_mix
            _edit_jobs[job_id]['approved_mix'] = None
            _edit_jobs[job_id]['_approve_event'] = approve_event

        # Wait up to 10 minutes for user approval
        approved = approve_event.wait(timeout=600)

        with _edit_jobs_lock:
            final_mix = _edit_jobs[job_id].get('approved_mix') or recommended_mix
            _edit_jobs[job_id].pop('_approve_event', None)

        if not approved:
            print(f"Enhance approval timed out for {job_id}, using recommended mix: {recommended_mix}")
        else:
            print(f"Enhance approved for {job_id}, mix: {final_mix}")

        # Enhance with Adobe Enhance Speech (before merging intro/outro)
        current_step = 'enhancing'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'enhancing'
        enhanced_path = process_with_adobe_enhance(wav_path, enhance_mix=final_mix)

        # Merge intro/outro after enhance — intro/outro are already produced audio
        if intro_path or outro_path:
            current_step = 'merging'
            with _edit_jobs_lock:
                _edit_jobs[job_id]['status'] = 'merging'
            enhanced_path = _concat_with_crossfade(enhanced_path, intro_path=intro_path, outro_path=outro_path)

        # Mastering: click removal + peak limiting + loudness normalization + MP3
        current_step = 'mastering'
        with _edit_jobs_lock:
            _edit_jobs[job_id]['status'] = 'mastering'
        _finalize_audio(job_id, enhanced_path)

        # Clean up raw multi-track files and predetection cache
        if transcript_id:
            _cleanup_multitrack_files(transcript_id)
            with _predetection_cache_lock:
                _predetection_cache.pop(transcript_id, None)

    except Exception as e:
        import traceback
        traceback.print_exc()
        if transcript_id:
            _cleanup_multitrack_files(transcript_id)
            with _predetection_cache_lock:
                _predetection_cache.pop(transcript_id, None)
        with _edit_jobs_lock:
            _edit_jobs[job_id] = {
                'status': 'error',
                'error': str(e),
                'failed_step': current_step,
            }


def _review_audio(audio_path):
    """
    Analyze final audio for quality metrics using ffmpeg.
    Pass 1: ebur128 for loudness + true peak.
    Pass 2: silencedetect for unexpected silence gaps.
    Returns dict with passed, issues, and metrics.
    """
    issues = []
    metrics = {'integrated_lufs': 0.0, 'true_peak_dbtp': 0.0, 'loudness_range_lu': 0.0, 'duration_s': 0.0}

    # Pass 1: ebur128 loudness analysis
    result = subprocess.run(
        ['ffmpeg', '-hide_banner', '-i', audio_path,
         '-af', 'ebur128=peak=true', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=120,
    )
    stderr = result.stderr

    # Parse Summary block from ebur128 output
    summary_start = stderr.rfind('Summary:')
    if summary_start >= 0:
        summary = stderr[summary_start:]
        m = re.search(r'I:\s*([-\d.]+)\s*LUFS', summary)
        if m:
            metrics['integrated_lufs'] = float(m.group(1))
        m = re.search(r'LRA:\s*([-\d.]+)\s*LU', summary)
        if m:
            metrics['loudness_range_lu'] = float(m.group(1))
        m = re.search(r'Peak:\s*([-\d.]+)\s*dBFS', summary)
        if m:
            metrics['true_peak_dbtp'] = float(m.group(1))

    # Get duration
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', audio_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        metrics['duration_s'] = round(float(probe.stdout.strip()), 1)
    except (ValueError, AttributeError):
        pass

    # Check loudness range (after mastering, should be close to -16 LUFS)
    lufs = metrics['integrated_lufs']
    if lufs < -20 or lufs > -12:
        issues.append({
            'type': 'loudness',
            'severity': 'warning',
            'time_s': None,
            'description': f'Integrated loudness {lufs:.1f} LUFS is outside target range (-20 to -12)',
        })

    # Check true peak
    peak = metrics['true_peak_dbtp']
    if peak > -0.5:
        issues.append({
            'type': 'clipping',
            'severity': 'warning',
            'time_s': None,
            'description': f'True peak {peak:.1f} dBTP exceeds -0.5 dBTP ceiling',
        })

    # Pass 2: silencedetect for gaps > 500ms
    result = subprocess.run(
        ['ffmpeg', '-hide_banner', '-i', audio_path,
         '-af', 'silencedetect=noise=-40dB:d=0.5', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=120,
    )
    silence_events = []
    current_start = None
    for line in result.stderr.split('\n'):
        m = re.search(r'silence_start:\s*([\d.]+)', line)
        if m:
            current_start = float(m.group(1))
        m = re.search(r'silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)', line)
        if m and current_start is not None:
            silence_end = float(m.group(1))
            silence_dur = float(m.group(2))
            silence_events.append((current_start, silence_end, silence_dur))
            current_start = None

    # Filter: ignore silence in first/last 2 seconds (natural episode boundaries)
    duration_s = metrics['duration_s']
    for start, end, dur in silence_events:
        if start < 2.0 or (duration_s > 0 and end > duration_s - 2.0):
            continue
        issues.append({
            'type': 'silence',
            'severity': 'info',
            'time_s': round(start, 1),
            'description': f'Silence gap {dur:.1f}s at {start:.1f}s',
        })

    passed = all(i['severity'] != 'warning' for i in issues)

    return {
        'passed': passed,
        'issues': issues,
        'metrics': metrics,
    }


def _finalize_audio(job_id, enhanced_path):
    """Shared finalization: mastering chain + MP3 encode."""
    mp3_path = enhanced_path.rsplit('.', 1)[0] + '_final.mp3'

    # Mastering chain: click removal → peak limiting → loudness normalization.
    # These are non-destructive — if there's nothing to fix, audio passes through unchanged.
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', enhanced_path, '-af',
         'adeclick=w=55:o=75,alimiter=limit=0.89:attack=5:release=50,loudnorm=I=-16:TP=-1:LRA=11',
         '-b:a', '320k',
         '-f', 'mp3', mp3_path],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise Exception(f"ffmpeg finalize failed: {result.stderr[-500:]}")

    size_kb = os.path.getsize(mp3_path) // 1024
    print(f"Finalize: {mp3_path} ({size_kb}KB, 320kbps MP3, mastered)")

    # Review step: analyze the final MP3 for quality metrics
    with _edit_jobs_lock:
        _edit_jobs[job_id]['status'] = 'reviewing'
    review = _review_audio(mp3_path)
    print(f"Review: passed={review['passed']}, "
          f"LUFS={review['metrics']['integrated_lufs']:.1f}, "
          f"peak={review['metrics']['true_peak_dbtp']:.1f} dBTP, "
          f"issues={len(review['issues'])}")

    with _edit_jobs_lock:
        _edit_jobs[job_id]['status'] = 'completed'
        _edit_jobs[job_id]['path'] = mp3_path
        _edit_jobs[job_id]['is_mp3'] = True
        _edit_jobs[job_id]['review'] = review
    print(f"Finalize job {job_id} complete: {mp3_path}")


# ============================================================================
# SHOW NOTES — on-demand episode summary via Claude
# ============================================================================

def generate_show_notes(transcript_data, custom_instructions=''):
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
    utt_text = "\n".join(utt_lines) if utt_lines else transcript_data.get('text', '')[:30000]

    duration_ms = transcript_data.get('audio_duration', 0)
    duration_str = format_timestamp(duration_ms) if duration_ms else 'unknown'

    custom_block = f"\n\nADDITIONAL INSTRUCTIONS FROM PRODUCER:\n{custom_instructions}" if custom_instructions.strip() else ''

    prompt = f"""You are a podcast producer writing show notes for an episode. Based on the transcript below, generate concise, well-structured show notes.

TRANSCRIPT:
{utt_text}

EPISODE DURATION: {duration_str}{custom_block}

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


def _run_show_notes_from_text_job(job_id, text, custom_instructions=''):
    """Background thread: generate show notes from raw transcript text."""
    try:
        transcript_data = {'text': text}
        notes = generate_show_notes(transcript_data, custom_instructions)
        with _jobs_lock:
            _jobs[job_id] = {'status': 'completed', 'result': {'show_notes': notes}}
        print(f"Show notes from text job {job_id} complete")
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
    """Upload multiple tracks, combine and start transcription."""
    if request.method == 'OPTIONS':
        return '', 204

    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500

    # Collect track files and labels from form data
    track_paths = []
    labels = []
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
        args=(job_id, track_paths, labels),
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


@app.route('/api/filler-assessment/<transcript_id>', methods=['GET'])
def filler_assessment(transcript_id):
    """Lightweight filler stats from a completed transcript — called before Claude analysis."""
    if not ASSEMBLYAI_API_KEY:
        return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500
    try:
        transcript_data = get_transcription(transcript_id)
        status = transcript_data.get("status")
        if status != "completed":
            return jsonify({"error": f"Transcription not ready (status: {status})"}), 400

        words = transcript_data.get("words", [])
        fillers = _find_fillers(words)
        duration_ms = transcript_data.get("audio_duration", 0)
        duration_min = duration_ms / 60000 if duration_ms else 0

        # Breakdown by filler type
        by_type = {}
        for f in fillers:
            t = f["text"]
            by_type[t] = by_type.get(t, 0) + 1

        fillers_per_min = (len(fillers) / duration_min) if duration_min > 0 else 0

        # Suggestion based on density
        if fillers_per_min < 1:
            suggested_pct = 40
        elif fillers_per_min < 3:
            suggested_pct = 60
        elif fillers_per_min < 5:
            suggested_pct = 75
        else:
            suggested_pct = 85

        return jsonify({
            "total_fillers": len(fillers),
            "fillers_per_minute": round(fillers_per_min, 1),
            "duration_minutes": round(duration_min, 1),
            "by_type": by_type,
            "suggested_pct": suggested_pct,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _run_analysis_job(job_id, transcript_data, filename, preset_name, transcript_id, custom_instructions, is_multitrack=False, filler_pct_override=None):
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

        # Apply user's filler_pct override if provided
        if filler_pct_override is not None:
            preset_cfg = {**preset_cfg, 'filler_pct': filler_pct_override}

        with _jobs_lock:
            _jobs[job_id]['status'] = 'analyzing'

        edit_analysis = analyze_transcript_with_claude(transcript_data, preset_cfg, custom_instructions, is_multitrack=is_multitrack)
        report = generate_edit_report(filename, transcript_data, edit_analysis, preset_cfg)
        cuts_count = sum(1 for d in edit_analysis['edit_decisions'] if 'start_ms' in d and 'end_ms' in d)

        # Cache pre-detection results so audio editing doesn't re-run them
        predetection = edit_analysis.get('predetection')
        if predetection and transcript_id:
            with _predetection_cache_lock:
                _predetection_cache[transcript_id] = predetection
                print(f"Cached pre-detection for {transcript_id}: "
                      f"{len(predetection.get('stumbles', []))} stumbles, "
                      f"{len(predetection.get('stutters', []))} stutters, "
                      f"{len(predetection.get('fillers', []))} fillers, "
                      f"{len(predetection.get('meta_comments', []))} meta")

        # Filler stats: total detected vs removed by Claude
        all_fillers = predetection.get('fillers', []) if predetection else _find_fillers(transcript_data.get('words', []))
        filler_removed = sum(1 for d in edit_analysis['edit_decisions'] if d.get('type') == 'Remove Filler')
        filler_pct_used = preset_cfg.get('filler_pct', 60)

        result = {
            "success": True,
            "edit_decisions": edit_analysis['edit_decisions'],
            "cuts_count": cuts_count,
            "report": report,
            "filler_stats": {
                "total": len(all_fillers),
                "removed": filler_removed,
                "pct_used": filler_pct_used,
            },
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
        filler_pct = data.get('filler_pct')  # optional override from filler assessment

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
            args=(job_id, transcript_data, filename, preset_name, transcript_id, custom_instructions, is_multitrack, filler_pct),
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


@app.route('/api/show-notes-from-text', methods=['POST', 'OPTIONS'])
def show_notes_from_text():
    """Generate show notes from raw transcript text (no audio upload needed)."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        if not CLAUDE_API_KEY:
            return jsonify({"error": "CLAUDE_API_KEY not configured"}), 500
        data = request.json
        text = data.get('transcript_text', '').strip()
        if not text:
            return jsonify({"error": "No transcript_text provided"}), 400
        custom_instructions = data.get('custom_instructions', '')

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {'status': 'pending'}

        threading.Thread(
            target=_run_show_notes_from_text_job,
            args=(job_id, text, custom_instructions),
            daemon=True,
        ).start()

        return jsonify({"success": True, "job_id": job_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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

        # Parse multitrack flag
        is_multitrack = False
        if request.content_type and 'multipart' in request.content_type:
            is_multitrack = request.form.get('is_multitrack') == 'true'
        else:
            is_multitrack = data.get('is_multitrack', False)

        print(f"Edit job: {len(cuts_ms)} cuts, intro={'yes' if intro_path else 'no'}, outro={'yes' if outro_path else 'no'}, is_multitrack={is_multitrack}")

        with _edit_jobs_lock:
            _edit_jobs[job_id] = {'status': 'pending'}

        if not get_adobe_token():
            return jsonify({"error": "ADOBE_ENHANCE_TOKEN is required for audio editing"}), 400
        threading.Thread(
            target=_run_edit_job,
            args=(job_id, audio_path, cuts_ms, transcript_id),
            kwargs={'intro_path': intro_path, 'outro_path': outro_path},
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
        resp = {"status": "completed", "is_mp3": job.get('is_mp3', False)}
        if 'review' in job:
            resp['review'] = job['review']
        return jsonify(resp)
    if job['status'] == 'error':
        return jsonify({"error": job['error'], "failed_step": job.get('failed_step')}), 500
    # cutting, awaiting_enhance, enhancing, merging, mastering, reviewing, completed
    result = {"status": job['status']}
    if 'progress' in job:
        result['progress'] = job['progress']
    if job['status'] == 'awaiting_enhance':
        if 'analysis' in job:
            result['analysis'] = job['analysis']
        if 'recommended_mix' in job:
            result['recommended_mix'] = job['recommended_mix']
    return jsonify(result)


@app.route('/api/edit-audio-approve/<job_id>', methods=['POST', 'OPTIONS'])
def edit_audio_approve(job_id):
    """Accept enhance mix settings and resume the edit pipeline."""
    if request.method == 'OPTIONS':
        return '', 204
    data = request.json or {}
    mix = data.get('enhance_mix')
    with _edit_jobs_lock:
        job = _edit_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        if job.get('status') != 'awaiting_enhance':
            return jsonify({"error": f"Job not awaiting approval (status: {job.get('status')})"}), 400
        job['approved_mix'] = mix
        event = job.get('_approve_event')
    if event:
        event.set()
    return jsonify({"success": True})


@app.route('/api/edit-audio-download/<job_id>', methods=['GET'])
def edit_audio_download(job_id):
    with _edit_jobs_lock:
        job = _edit_jobs.get(job_id)
    if not job:
        return jsonify({"error": "File not ready"}), 404
    if job['status'] == 'completed':
        return send_file(job['path'], as_attachment=True, download_name='edited_podcast.mp3', mimetype='audio/mpeg')
    return jsonify({"error": "File not ready"}), 404


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


# ============================================================================
# GOOGLE DOCS EXPORT
# ============================================================================

def _markdown_to_docs_requests(markdown_text):
    """Convert markdown show notes to Google Docs API batch update requests."""
    requests_list = []
    idx = 1  # cursor position (1-based, after initial newline)

    for line in markdown_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        # Determine heading level or bullet
        heading_level = None
        text = stripped
        is_bullet = False

        if stripped.startswith('### '):
            heading_level = 'HEADING_3'
            text = stripped[4:]
        elif stripped.startswith('## '):
            heading_level = 'HEADING_2'
            text = stripped[3:]
        elif stripped.startswith('# '):
            heading_level = 'HEADING_1'
            text = stripped[2:]
        elif stripped.startswith('- '):
            is_bullet = True
            text = stripped[2:]

        # Strip bold markers for plain text insertion
        clean_text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        insert_text = clean_text + '\n'

        # Insert text
        requests_list.append({
            'insertText': {
                'location': {'index': idx},
                'text': insert_text,
            }
        })

        # Apply heading style
        if heading_level:
            requests_list.append({
                'updateParagraphStyle': {
                    'range': {'startIndex': idx, 'endIndex': idx + len(insert_text)},
                    'paragraphStyle': {'namedStyleType': heading_level},
                    'fields': 'namedStyleType',
                }
            })

        # Apply bullet
        if is_bullet:
            requests_list.append({
                'createParagraphBullets': {
                    'range': {'startIndex': idx, 'endIndex': idx + len(insert_text)},
                    'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE',
                }
            })

        # Apply bold to **text** segments
        bold_pattern = re.compile(r'\*\*(.+?)\*\*')
        offset = 0
        for match in bold_pattern.finditer(text):
            # Calculate position in clean text
            bold_text = match.group(1)
            # Find where this bold text is in clean_text
            bold_start = clean_text.find(bold_text, offset)
            if bold_start >= 0:
                requests_list.append({
                    'updateTextStyle': {
                        'range': {
                            'startIndex': idx + bold_start,
                            'endIndex': idx + bold_start + len(bold_text),
                        },
                        'textStyle': {'bold': True},
                        'fields': 'bold',
                    }
                })
                offset = bold_start + len(bold_text)

        idx += len(insert_text)

    return requests_list


@app.route('/api/export-google-doc', methods=['POST', 'OPTIONS'])
def export_google_doc():
    """Create a Google Doc from show notes markdown."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        data = request.json
        show_notes = data.get('show_notes', '').strip()
        title = data.get('title', 'Show Notes')
        if not show_notes:
            return jsonify({"error": "No show_notes provided"}), 400

        if not _GOOGLE_REFRESH_TOKEN:
            return jsonify({"error": "GOOGLE_REFRESH_TOKEN not configured"}), 500

        creds = Credentials(
            token=None,
            refresh_token=_GOOGLE_REFRESH_TOKEN,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=os.environ.get('GOOGLE_CLIENT_ID', ''),
            client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', ''),
        )

        docs_service = build('docs', 'v1', credentials=creds)

        # Create empty doc
        doc = docs_service.documents().create(body={'title': title}).execute()
        doc_id = doc['documentId']
        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

        # Build formatting requests
        reqs = _markdown_to_docs_requests(show_notes)
        if reqs:
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': reqs},
            ).execute()

        print(f"Created Google Doc: {doc_url}")
        return jsonify({"success": True, "url": doc_url, "doc_id": doc_id})
    except ImportError:
        return jsonify({"error": "Google API libraries not installed"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/export-transcript-google-doc/<transcript_id>', methods=['POST', 'OPTIONS'])
def export_transcript_google_doc(transcript_id):
    """Create a Google Doc from a transcript."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        if not ASSEMBLYAI_API_KEY:
            return jsonify({"error": "ASSEMBLYAI_API_KEY not configured"}), 500
        if not _GOOGLE_REFRESH_TOKEN:
            return jsonify({"error": "GOOGLE_REFRESH_TOKEN not configured"}), 500

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

        creds = Credentials(
            token=None,
            refresh_token=_GOOGLE_REFRESH_TOKEN,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=os.environ.get('GOOGLE_CLIENT_ID', ''),
            client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', ''),
        )

        docs_service = build('docs', 'v1', credentials=creds)
        doc = docs_service.documents().create(body={'title': 'Transcript'}).execute()
        doc_id = doc['documentId']
        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

        # Insert transcript text
        if transcript_text:
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': [{'insertText': {'location': {'index': 1}, 'text': transcript_text}}]},
            ).execute()

        print(f"Created Transcript Google Doc: {doc_url}")
        return jsonify({"success": True, "url": doc_url, "doc_id": doc_id})
    except ImportError:
        return jsonify({"error": "Google API libraries not installed"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
