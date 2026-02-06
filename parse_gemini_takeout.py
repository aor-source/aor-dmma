#!/usr/bin/env python3
"""
Gemini Takeout Parser
=====================
Parses Google Takeout export of Gemini conversations
and extracts AoR content.

Usage:
    python3 parse_gemini_takeout.py ~/Downloads/Takeout/
    python3 parse_gemini_takeout.py ~/Downloads/takeout-*.zip

Authors: Jon + Claude - February 2026
"""

import os
import sys
import json
import re
import zipfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict

OUTPUT_DIR = Path.home() / "aor-dmma" / "aor_catalog"

# AoR detection patterns
VERSE_PATTERNS = [
    r'\[Verse\s*\d*\]',
    r'\[Hook\]',
    r'\[Chorus\]',
    r'\[Bridge\]',
    r'\[Intro\]',
    r'\[Outro\]',
]

SIGNATURE_PATTERNS = [
    r'I am the Architect of Rhyme',
    r'— Architect of Rhyme',
    r'- AoR',
    r'\[The Architect\]',
]


def is_aor_content(text: str) -> bool:
    """Check if text contains AoR creative content"""
    if not text:
        return False

    for pattern in VERSE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    for pattern in SIGNATURE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Check for lyrical structure
    lines = text.strip().split('\n')
    short_lines = [l for l in lines if 10 < len(l.strip()) < 80]
    if len(short_lines) > 8:
        return True

    return False


def parse_structure(text: str) -> dict:
    """Parse song structure into components"""
    structure = {
        "verses": [],
        "hooks": [],
        "bridges": [],
        "other": [],
    }

    sections = re.split(
        r'(\[(?:Verse|Hook|Chorus|Bridge|Intro|Outro|Pre-Chorus)\s*\d*\])',
        text, flags=re.IGNORECASE
    )

    current_section = "intro"
    current_content = []

    for part in sections:
        part = part.strip()
        if not part:
            continue

        marker_match = re.match(
            r'\[(Verse|Hook|Chorus|Bridge|Intro|Outro|Pre-Chorus)\s*(\d*)\]',
            part, re.IGNORECASE
        )
        if marker_match:
            if current_content:
                content = '\n'.join(current_content).strip()
                if 'verse' in current_section.lower():
                    structure["verses"].append(content)
                elif 'hook' in current_section.lower() or 'chorus' in current_section.lower():
                    structure["hooks"].append(content)
                elif 'bridge' in current_section.lower():
                    structure["bridges"].append(content)
                else:
                    structure["other"].append(content)

            current_section = marker_match.group(1)
            current_content = []
        else:
            current_content.append(part)

    if current_content:
        content = '\n'.join(current_content).strip()
        if 'verse' in current_section.lower():
            structure["verses"].append(content)
        elif 'hook' in current_section.lower() or 'chorus' in current_section.lower():
            structure["hooks"].append(content)
        elif 'bridge' in current_section.lower():
            structure["bridges"].append(content)
        else:
            structure["other"].append(content)

    return structure


def extract_title(text: str) -> str:
    """Extract or infer title from content"""
    patterns = [
        r'\[Title:\s*([^\]]+)\]',
        r'\*\*([^*]+)\*\*',
        r'"([^"]+)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('[')]
    if lines:
        return f"Untitled ({lines[0][:40]}...)"

    return "Untitled"


def count_bars(text: str) -> int:
    """Estimate bar count"""
    lines = text.strip().split('\n')
    bar_count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('—'):
            continue
        if 10 < len(line) < 150:
            bar_count += 1
    return bar_count


def parse_gemini_json(filepath: Path) -> list:
    """Parse a Gemini conversation JSON file"""
    entries = []

    try:
        with open(filepath, 'r', errors='ignore') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return entries

    # Gemini Takeout format varies - handle common structures
    conversations = []

    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict):
        if 'conversations' in data:
            conversations = data['conversations']
        elif 'messages' in data:
            conversations = [data]
        else:
            conversations = [data]

    for conv in conversations:
        # Extract messages
        messages = []
        if isinstance(conv, dict):
            messages = conv.get('messages', conv.get('turns', conv.get('content', [])))
            if not isinstance(messages, list):
                messages = [conv]

        for msg in messages:
            # Get text content
            text = ""
            if isinstance(msg, str):
                text = msg
            elif isinstance(msg, dict):
                text = msg.get('text', msg.get('content', msg.get('response', '')))
                if isinstance(text, list):
                    text = '\n'.join(str(t) for t in text)

                # Check if this is model response
                role = msg.get('role', msg.get('author', '')).lower()
                if role and 'user' in role:
                    continue

            if not text or not is_aor_content(text):
                continue

            structure = parse_structure(text)
            title = extract_title(text)

            # Get timestamp
            timestamp = ""
            if isinstance(conv, dict):
                timestamp = conv.get('createTime', conv.get('timestamp', conv.get('date', '')))
            if isinstance(msg, dict):
                timestamp = msg.get('createTime', msg.get('timestamp', timestamp))

            entries.append({
                'id': f"gemini_{filepath.stem}_{len(entries)}",
                'title': title,
                'timestamp': timestamp,
                'source': 'gemini_takeout',
                'source_file': filepath.name,
                'structure': {
                    'verse_count': len(structure['verses']),
                    'has_hook': len(structure['hooks']) > 0,
                    'has_bridge': len(structure['bridges']) > 0,
                    'bar_count': count_bars(text),
                },
                'verses': structure['verses'],
                'hooks': structure['hooks'],
                'bridges': structure['bridges'],
                'raw_text': text[:5000],
            })

    return entries


def parse_gemini_html(filepath: Path) -> list:
    """Parse Gemini HTML export"""
    entries = []

    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read()
    except:
        return entries

    # Simple HTML parsing - look for response blocks
    import html
    from html.parser import HTMLParser

    class GeminiHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_response = False
            self.current_text = []
            self.responses = []

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            classes = attrs_dict.get('class', '')
            if 'response' in classes or 'model' in classes or 'assistant' in classes:
                self.in_response = True
                self.current_text = []

        def handle_endtag(self, tag):
            if self.in_response and tag in ('div', 'section', 'article'):
                text = ' '.join(self.current_text)
                if text:
                    self.responses.append(text)
                self.in_response = False
                self.current_text = []

        def handle_data(self, data):
            if self.in_response:
                self.current_text.append(data.strip())

    parser = GeminiHTMLParser()
    try:
        parser.feed(content)
    except:
        pass

    for i, text in enumerate(parser.responses):
        if not is_aor_content(text):
            continue

        structure = parse_structure(text)
        entries.append({
            'id': f"gemini_html_{filepath.stem}_{i}",
            'title': extract_title(text),
            'timestamp': '',
            'source': 'gemini_takeout_html',
            'source_file': filepath.name,
            'structure': {
                'verse_count': len(structure['verses']),
                'has_hook': len(structure['hooks']) > 0,
                'has_bridge': len(structure['bridges']) > 0,
                'bar_count': count_bars(text),
            },
            'verses': structure['verses'],
            'hooks': structure['hooks'],
            'bridges': structure['bridges'],
            'raw_text': text[:5000],
        })

    return entries


def process_takeout(takeout_path: Path) -> list:
    """Process Takeout export (folder or zip)"""
    all_entries = []
    stats = defaultdict(int)

    # Handle zip file
    if takeout_path.suffix == '.zip':
        print(f"[*] Extracting zip: {takeout_path.name}")
        extract_dir = takeout_path.parent / takeout_path.stem
        with zipfile.ZipFile(takeout_path, 'r') as zf:
            zf.extractall(extract_dir)
        takeout_path = extract_dir

    # Find Gemini data
    gemini_dirs = [
        takeout_path / "Gemini Apps",
        takeout_path / "Takeout" / "Gemini Apps",
        takeout_path / "Bard",
        takeout_path / "Takeout" / "Bard",
        takeout_path,  # Direct folder
    ]

    found_dir = None
    for gd in gemini_dirs:
        if gd.exists():
            found_dir = gd
            break

    if not found_dir:
        print(f"[!] Could not find Gemini data in {takeout_path}")
        print("    Expected: Gemini Apps/ or Bard/ folder")
        return []

    print(f"[*] Processing: {found_dir}")

    # Process all JSON and HTML files
    for ext in ['*.json', '*.html']:
        for filepath in found_dir.rglob(ext):
            stats['files'] += 1
            print(f"    Processing {filepath.name}...")

            if filepath.suffix == '.json':
                entries = parse_gemini_json(filepath)
            else:
                entries = parse_gemini_html(filepath)

            if entries:
                print(f"      → Found {len(entries)} AoR tracks")
                all_entries.extend(entries)
                stats['tracks'] += len(entries)

    print(f"\n[+] Takeout processing complete")
    print(f"    Files processed: {stats['files']}")
    print(f"    AoR tracks found: {stats['tracks']}")

    return all_entries


def save_catalog(entries: list):
    """Save extracted entries"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"aor_gemini_{timestamp}.jsonl"

    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n[+] Saved: {output_path}")
    print(f"    Tracks: {len(entries)}")

    # Stats
    total_bars = sum(e.get('structure', {}).get('bar_count', 0) for e in entries)
    stats = {
        'generated': datetime.now().isoformat(),
        'source': 'gemini_takeout',
        'total_tracks': len(entries),
        'total_bars': total_bars,
    }

    stats_path = OUTPUT_DIR / "gemini_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"[+] Stats: {stats_path}")
    print(f"\nNow run: python3 ~/aor-dmma/merge_catalogs.py")

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_gemini_takeout.py <takeout-path>")
        print("")
        print("  takeout-path: Path to Takeout folder or .zip file")
        print("")
        print("To get Takeout:")
        print("  1. Go to https://takeout.google.com")
        print("  2. Deselect all, then select 'Gemini Apps'")
        print("  3. Export and download")
        return

    takeout_path = Path(sys.argv[1]).expanduser().resolve()
    if not takeout_path.exists():
        print(f"[!] Path not found: {takeout_path}")
        return

    entries = process_takeout(takeout_path)
    if entries:
        save_catalog(entries)
    else:
        print("[!] No AoR content found in Takeout export")


if __name__ == "__main__":
    main()
