#!/usr/bin/env python3
"""
AoR File Extractor
==================
Extracts AoR content from already-extracted files (docx, pdf, txt).
Works on the OrganizedTakeout directory structure.

Usage:
    python3 extract_aor_files.py                    # Process all
    python3 extract_aor_files.py /path/to/folder

Authors: Jon + Claude - February 2026
"""

import os
import sys
import json
import re
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from xml.etree import ElementTree as ET

OUTPUT_DIR = Path.home() / "aor-dmma" / "aor_catalog"

# Directories with extracted files
SOURCE_DIRS = [
    Path.home() / "Documents/OrganizedTakeout/Unique_PDFs_Only",
    Path.home() / "Documents/OrganizedTakeout/Takeout-5",
    Path.home() / "Documents/OrganizedTakeout/Takeout 5",
    Path.home() / "Documents/OrganizedTakeout/Converted_PDFs",
]

# AoR detection patterns
AOR_PATTERNS = [
    r'\[Verse\s*\d*\]',
    r'\[Hook\]',
    r'\[Chorus\]',
    r'\[Bridge\]',
    r'\[Intro\]',
    r'\[Outro\]',
    r'Architect of Rhyme',
    r'— AoR',
    r'- AoR',
    r'I am the Architect',
]

AOR_FILENAME_PATTERNS = [
    r'architect.*rhyme',
    r'aor[_\s\-]',
    r'hip\s*hop',
    r'EP\s*\d',
    r'verse',
    r'track',
    r'album',
    r'bars',
    r'lyrics',
    r'rhyme',
    r'gemini',
    r'conversation',
]


def is_aor_filename(name: str) -> bool:
    """Check if filename suggests AoR content"""
    name_lower = name.lower()
    return any(re.search(p, name_lower, re.IGNORECASE) for p in AOR_FILENAME_PATTERNS)


def is_aor_content(text: str) -> bool:
    """Check if text contains AoR creative content"""
    if not text:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in AOR_PATTERNS)


def extract_docx_text(filepath: Path) -> str:
    """Extract text from a .docx file"""
    text_parts = []

    try:
        with zipfile.ZipFile(filepath) as docx:
            if 'word/document.xml' in docx.namelist():
                xml_content = docx.read('word/document.xml')
                tree = ET.fromstring(xml_content)

                for elem in tree.iter():
                    if elem.tag.endswith('}t'):
                        if elem.text:
                            text_parts.append(elem.text)
                    elif elem.tag.endswith('}p'):
                        text_parts.append('\n')
    except Exception as e:
        print(f"      [!] Error reading docx: {e}")

    return ''.join(text_parts)


def extract_pdf_text(filepath: Path) -> str:
    """Extract text from PDF using system tools"""
    text = ""

    # Try pdftotext (poppler)
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', str(filepath), '-'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            text = result.stdout
    except FileNotFoundError:
        pass
    except Exception as e:
        pass

    # Fallback: try textutil (macOS)
    if not text:
        try:
            result = subprocess.run(
                ['textutil', '-convert', 'txt', '-stdout', str(filepath)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                text = result.stdout
        except:
            pass

    return text


def parse_structure(text: str) -> dict:
    """Parse song structure into components"""
    structure = {"verses": [], "hooks": [], "bridges": [], "other": []}

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


def extract_title(text: str, filename: str) -> str:
    """Extract or infer title"""
    patterns = [
        r'\[Title:\s*([^\]]+)\]',
        r'\*\*([^*]+)\*\*',
    ]

    for pattern in patterns:
        match = re.search(pattern, text[:500])
        if match:
            return match.group(1).strip()

    # Clean up filename
    name = Path(filename).stem
    # Remove hash suffixes like _292c
    name = re.sub(r'[-_][a-f0-9]{4,}$', '', name)
    name = re.sub(r'[_-]', ' ', name)
    return name[:80]


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


def process_directory(source_dir: Path) -> list:
    """Process all AoR files in a directory"""
    entries = []
    stats = defaultdict(int)

    if not source_dir.exists():
        return entries

    print(f"[*] Processing: {source_dir}")

    # Find relevant files
    files = []
    for ext in ['*.docx', '*.pdf', '*.txt']:
        files.extend(source_dir.glob(ext))

    # Filter to AoR-related filenames
    aor_files = [f for f in files if is_aor_filename(f.name)]
    print(f"    Found {len(aor_files)} files with AoR-related names")

    for filepath in aor_files:
        stats['checked'] += 1

        # Extract text based on file type
        text = ""
        if filepath.suffix == '.docx':
            text = extract_docx_text(filepath)
        elif filepath.suffix == '.pdf':
            text = extract_pdf_text(filepath)
        elif filepath.suffix == '.txt':
            try:
                text = filepath.read_text(errors='ignore')
            except:
                pass

        if not text:
            stats['empty'] += 1
            continue

        # Verify it has AoR content
        if not is_aor_content(text):
            stats['no_aor'] += 1
            continue

        stats['found'] += 1

        structure = parse_structure(text)
        title = extract_title(text, filepath.name)

        entries.append({
            'id': f"files_{filepath.stem}",
            'title': title,
            'timestamp': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
            'source': 'extracted_files',
            'source_file': filepath.name,
            'source_dir': source_dir.name,
            'structure': {
                'verse_count': len(structure['verses']),
                'has_hook': len(structure['hooks']) > 0,
                'has_bridge': len(structure['bridges']) > 0,
                'bar_count': count_bars(text),
            },
            'verses': structure['verses'],
            'hooks': structure['hooks'],
            'bridges': structure['bridges'],
            'raw_text': text[:10000],
        })

        print(f"    ✓ {title}")

    print(f"    → Found {stats['found']} AoR tracks")
    return entries


def save_catalog(entries: list):
    """Save extracted entries"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"aor_files_{timestamp}.jsonl"

    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n[+] Saved: {output_path}")
    print(f"    Tracks: {len(entries)}")

    # Stats
    total_bars = sum(e.get('structure', {}).get('bar_count', 0) for e in entries)
    total_verses = sum(len(e.get('verses', [])) for e in entries)

    stats = {
        'generated': datetime.now().isoformat(),
        'source': 'extracted_files',
        'total_tracks': len(entries),
        'total_verses': total_verses,
        'total_bars': total_bars,
    }

    stats_path = OUTPUT_DIR / "files_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return output_path


def main():
    dirs_to_process = []

    # Check for specific directory argument
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            continue
        p = Path(arg).expanduser().resolve()
        if p.exists() and p.is_dir():
            dirs_to_process.append(p)

    # Default to known source directories
    if not dirs_to_process:
        dirs_to_process = [d for d in SOURCE_DIRS if d.exists()]

    if not dirs_to_process:
        print("[!] No directories found to process")
        return

    all_entries = []
    for source_dir in dirs_to_process:
        entries = process_directory(source_dir)
        all_entries.extend(entries)

    if all_entries:
        save_catalog(all_entries)
        print(f"\nNow run: python3 ~/aor-dmma/merge_catalogs.py")
    else:
        print("[!] No AoR content found")


if __name__ == "__main__":
    main()
