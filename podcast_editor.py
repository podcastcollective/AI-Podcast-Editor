"""
AI Podcast Editor - Main Processing Script
Integrates AssemblyAI for transcription and Claude API for intelligent editing decisions
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

# You'll need to install these packages:
# pip install anthropic requests
import anthropic
import requests


class PodcastEditor:
    def __init__(self, assemblyai_key: str, claude_key: str):
        """Initialize the podcast editor with API keys"""
        self.assemblyai_key = assemblyai_key
        self.claude_client = anthropic.Anthropic(api_key=claude_key)
        self.assemblyai_base_url = "https://api.assemblyai.com/v2"
        
    def upload_to_assemblyai(self, audio_file_path: str) -> str:
        """Upload audio file to AssemblyAI and return upload URL"""
        print("📤 Uploading audio to AssemblyAI...")
        
        headers = {"authorization": self.assemblyai_key}
        
        with open(audio_file_path, "rb") as f:
            response = requests.post(
                f"{self.assemblyai_base_url}/upload",
                headers=headers,
                data=f
            )
        
        if response.status_code == 200:
            upload_url = response.json()["upload_url"]
            print(f"✅ Upload successful: {upload_url}")
            return upload_url
        else:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
    
    def transcribe_audio(self, audio_url: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Transcribe audio using AssemblyAI with speaker detection"""
        print("🎙️  Starting transcription with AssemblyAI...")
        
        headers = {
            "authorization": self.assemblyai_key,
            "content-type": "application/json"
        }
        
        # Default configuration with speaker detection
        transcription_config = {
            "audio_url": audio_url,
            "speaker_labels": True,  # Enable speaker diarization
            "punctuate": True,
            "format_text": True,
        }
        
        # Add custom config if provided
        if config:
            transcription_config.update(config)
        
        response = requests.post(
            f"{self.assemblyai_base_url}/transcript",
            json=transcription_config,
            headers=headers
        )
        
        transcript_id = response.json()["id"]
        print(f"📝 Transcription job created: {transcript_id}")
        
        # Poll for completion
        polling_url = f"{self.assemblyai_base_url}/transcript/{transcript_id}"
        
        while True:
            response = requests.get(polling_url, headers=headers)
            status = response.json()["status"]
            
            if status == "completed":
                print("✅ Transcription complete!")
                return response.json()
            elif status == "error":
                raise Exception(f"Transcription failed: {response.json()['error']}")
            
            print(f"⏳ Status: {status}... waiting 5 seconds")
            time.sleep(5)
    
    def analyze_transcript_with_claude(
        self, 
        transcript_data: Dict[str, Any],
        requirements: Dict[str, bool],
        custom_instructions: str = ""
    ) -> Dict[str, Any]:
        """Use Claude to analyze transcript and generate intelligent edit decisions"""
        print("🤖 Analyzing transcript with Claude API...")
        
        # Extract utterances (speaker-segmented text)
        utterances = transcript_data.get("utterances", [])
        
        # Build transcript text with timestamps
        transcript_text = ""
        for utt in utterances[:100]:  # Limit for token efficiency in demo
            speaker = utt.get("speaker", "Unknown")
            start_time = self._format_timestamp(utt.get("start", 0))
            text = utt.get("text", "")
            transcript_text += f"[{start_time}] Speaker {speaker}: {text}\n"
        
        # Create the prompt for Claude
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

TASK:
Generate a detailed edit decision list (EDL) for this podcast. For each edit decision, provide:
1. Timestamp (format: HH:MM:SS)
2. Edit type (Remove Filler, Trim Pause, Keep Pause, Audio Fix, Content Cut)
3. Description of what to do
4. Confidence score (0-100)
5. Rationale for the decision

Focus on:
- Removing excessive filler words (um, uh, like, you know) while keeping natural speech
- Trimming long pauses (>2 seconds) to 1-1.5 seconds
- Preserving intentional pauses for comedic timing or emphasis
- Flagging any sections that need human review
- Maintaining the natural flow and energy of conversation

Return your response as a JSON array of edit decisions with this structure:
[
  {{
    "timestamp": "00:02:34",
    "type": "Remove Filler",
    "description": "Remove 'um, uh' (2 instances)",
    "confidence": 95,
    "rationale": "Consecutive filler words with no content value"
  }},
  ...
]

Only return the JSON array, no other text."""

        # Call Claude API
        message = self.claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from response
        response_text = message.content[0].text
        
        try:
            # Try to parse as JSON
            edit_decisions = json.loads(response_text)
        except json.JSONDecodeError:
            # If Claude included text around the JSON, extract it
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                edit_decisions = json.loads(json_match.group())
            else:
                raise Exception("Could not parse Claude's response as JSON")
        
        print(f"✅ Generated {len(edit_decisions)} edit decisions")
        return {
            "edit_decisions": edit_decisions,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def generate_edit_report(
        self,
        audio_file_path: str,
        transcript_data: Dict[str, Any],
        edit_analysis: Dict[str, Any],
        requirements: Dict[str, bool]
    ) -> str:
        """Generate a comprehensive edit report for human editors"""
        print("📋 Generating edit report...")
        
        report = f"""
================================================================================
                        AI PODCAST EDIT REPORT
================================================================================

EPISODE: {os.path.basename(audio_file_path)}
PROCESSED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
DURATION: {self._format_timestamp(transcript_data.get('audio_duration', 0))}

--------------------------------------------------------------------------------
CLIENT REQUIREMENTS
--------------------------------------------------------------------------------
✓ Remove filler words: {requirements.get('removeFillerWords', True)}
✓ Remove long pauses: {requirements.get('removeLongPauses', True)}
✓ Normalize audio: {requirements.get('normalizeAudio', True)}
✓ Remove background noise: {requirements.get('removeBackgroundNoise', True)}
Target length: {requirements.get('targetLength', 'Not specified')}

--------------------------------------------------------------------------------
TRANSCRIPT STATISTICS
--------------------------------------------------------------------------------
Total words: {transcript_data.get('words', []) and len(transcript_data.get('words', []))}
Speakers detected: {len(set(u.get('speaker') for u in transcript_data.get('utterances', [])))}
Confidence: {transcript_data.get('confidence', 0):.1%}

--------------------------------------------------------------------------------
EDIT DECISION LIST (EDL)
--------------------------------------------------------------------------------
Total edits: {len(edit_analysis['edit_decisions'])}

"""
        
        # Add each edit decision
        for i, decision in enumerate(edit_analysis['edit_decisions'], 1):
            report += f"""
[{i}] {decision['timestamp']} - {decision['type']}
    Description: {decision['description']}
    Confidence: {decision['confidence']}%
    Rationale: {decision['rationale']}
"""
        
        report += """
--------------------------------------------------------------------------------
HUMAN EDITOR CHECKLIST
--------------------------------------------------------------------------------
☐ Listen to full edited episode
☐ Verify no jarring cuts or audio artifacts
☐ Confirm natural speech flow maintained
☐ Check all client requirements met
☐ Validate audio levels are consistent
☐ Review all flagged sections

NOTES:
_____________________________________________________________________
_____________________________________________________________________
_____________________________________________________________________

EDITOR SIGNATURE: _________________ DATE: _____________

================================================================================
"""
        
        return report
    
    def _format_timestamp(self, milliseconds: int) -> str:
        """Convert milliseconds to HH:MM:SS format"""
        seconds = milliseconds / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def process_episode(
        self,
        audio_file_path: str,
        requirements: Dict[str, bool],
        custom_instructions: str = "",
        output_dir: str = "./output"
    ) -> Dict[str, str]:
        """
        Complete processing pipeline for a podcast episode
        
        Returns dict with paths to:
        - transcript_json: Full AssemblyAI transcript
        - edit_decisions_json: Claude's edit decisions
        - edit_report_txt: Human-readable report
        """
        print("\n" + "="*80)
        print("🎬 STARTING AI PODCAST EDITOR")
        print("="*80 + "\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Upload audio
        audio_url = self.upload_to_assemblyai(audio_file_path)
        
        # Step 2: Transcribe with speaker detection
        transcript_data = self.transcribe_audio(audio_url)
        
        # Save transcript
        transcript_path = os.path.join(output_dir, "transcript.json")
        with open(transcript_path, "w") as f:
            json.dump(transcript_data, f, indent=2)
        print(f"💾 Saved transcript to: {transcript_path}")
        
        # Step 3: Analyze with Claude
        edit_analysis = self.analyze_transcript_with_claude(
            transcript_data,
            requirements,
            custom_instructions
        )
        
        # Save edit decisions
        edit_decisions_path = os.path.join(output_dir, "edit_decisions.json")
        with open(edit_decisions_path, "w") as f:
            json.dump(edit_analysis, f, indent=2)
        print(f"💾 Saved edit decisions to: {edit_decisions_path}")
        
        # Step 4: Generate report
        report = self.generate_edit_report(
            audio_file_path,
            transcript_data,
            edit_analysis,
            requirements
        )
        
        report_path = os.path.join(output_dir, "edit_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"💾 Saved edit report to: {report_path}")
        
        print("\n" + "="*80)
        print("✅ PROCESSING COMPLETE!")
        print("="*80 + "\n")
        
        return {
            "transcript_json": transcript_path,
            "edit_decisions_json": edit_decisions_path,
            "edit_report_txt": report_path
        }


def main():
    """Example usage of the PodcastEditor"""
    
    # ============================================================================
    # CONFIGURATION - Replace with your actual API keys
    # ============================================================================
    ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "your_assemblyai_key_here")
    CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "your_claude_key_here")
    
    # Path to your podcast audio file
    AUDIO_FILE = "path/to/your/podcast.mp3"
    
    # Client requirements
    requirements = {
        "removeFillerWords": True,
        "removeLongPauses": True,
        "normalizeAudio": True,
        "removeBackgroundNoise": True,
        "targetLength": "45 minutes"
    }
    
    # Custom editing instructions
    custom_instructions = """
    Keep the banter between the hosts at the beginning.
    There's a technical difficulty around 15:30 - remove that section.
    Preserve the joke at 22:15, including the pause before the punchline.
    """
    
    # ============================================================================
    # RUN THE EDITOR
    # ============================================================================
    
    editor = PodcastEditor(ASSEMBLYAI_API_KEY, CLAUDE_API_KEY)
    
    try:
        results = editor.process_episode(
            audio_file_path=AUDIO_FILE,
            requirements=requirements,
            custom_instructions=custom_instructions,
            output_dir="./podcast_edits"
        )
        
        print("\n📂 Output files:")
        for key, path in results.items():
            print(f"   {key}: {path}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
