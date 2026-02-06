#!/usr/bin/env python3
"""
AoR-MIR Component 1: TRANSCRIPTION
===================================
Whisper-based audio transcription with word timestamps.
Outputs: {stem}_transcript.json

Requires: torch, openai-whisper
"""

import sys
import json
from pathlib import Path

def transcribe(audio_path: str, output_dir: str, model_size: str = "medium"):
    import torch
    import whisper

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🎤 Transcribing: {audio_path.name} on {device.upper()}")

    # Load model
    model = whisper.load_model(model_size, device=device)

    # Transcribe
    result = model.transcribe(str(audio_path), word_timestamps=True, language="en")

    # Extract data
    output = {
        "source_file": audio_path.name,
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "language": result.get("language", "en")
    }

    # Save
    out_file = output_dir / f"{audio_path.stem}_transcript.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved: {out_file}")
    return str(out_file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 01_transcribe.py <audio_file> <output_dir> [model_size]")
        sys.exit(1)

    model = sys.argv[3] if len(sys.argv) > 3 else "medium"
    transcribe(sys.argv[1], sys.argv[2], model)
