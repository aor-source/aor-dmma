#!/usr/bin/env python3
"""
Takeout AoR Extractor
=====================
Extracts AoR content from Google Takeout archives.
Handles .docx files (Word documents stored in Takeout).

Usage:
    python3 extract_takeout_aor.py                    # Process all known archives
    python3 extract_takeout_aor.py path/to/takeout.zip
    python3 extract_takeout_aor.py --list             # List archives and AoR file counts

Authors: Jon + Claude - February 2026
"""

import os
import sys
import json
import re
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from xml.etree import ElementTree as ET

OUTPUT_DIR = Path.home() / "aor-dmma" / "aor_catalog"
SCRATCHPAD = Path("/private/tmp/claude-501/-Users-alignmentnerd/1bd860fe-b9ac-4f48-a8e5-ce02355c4e52/scratchpad")

# Priority archives (highest AoR content)
PRIORITY_ARCHIVES = [
    Path.home() / "Documents/OrganizedTakeout/Takeout 5/takeout-20250927T082831Z-1-001.zip",
    Path.home() / "Documents/OrganizedTakeout/Takeout 10/takeout-20250910T081105Z-1-001.zip",
    Path.home() / "Documents/OrganizedTakeout/Takeout-2/takeout-20251126T124004Z-6-001.zip",
    Path.home() / "Documents/OrganizedTakeout/Takeout 5/takeout-20250910T081105Z-5-001.zip",
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
    r'\[The Architect\]',
]

AOR_FILENAME_PATTERNS = [
    r'architect.*rhyme',
    r'aor[_\s]',
    r'verse',
    r'hip\s*hop',
    r'bars',
    r'lyrics',
]


def is_aor_filename(name: str) -> bool:
    """Check if filename suggests AoR content"""
    name_lower = name.lower()
    return any(re.search(p, name_lower) for p in AOR_FILENAME_PATTERNS)


def is_aor_content(text: str) -> bool:
    """Check if text contains AoR creative content"""
    if not text:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in AOR_PATTERNS)


def extract_docx_text(docx_bytes: bytes) -> str:
    """Extract text from a .docx file (which is a ZIP containing XML)"""
    import io
    text_parts = []

    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as docx:
            # Main document content
            if 'word/document.xml' in docx.namelist():
                xml_content = docx.read('word/document.xml')
                tree = ET.fromstring(xml_content)

                # Word namespace
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

                # Extract all text elements
                for elem in tree.iter():
                    if elem.tag.endswith('}t'):  # Text element
                        if elem.text:
                            text_parts.append(elem.text)
                    elif elem.tag.endswith('}p'):  # Paragraph - add newline
                        text_parts.append('\n')
    except Exception as e:
        pass

    return ''.join(text_parts)


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
        r'"([^"]+)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, text[:500])
        if match:
            return match.group(1).strip()

    # Use filename
    name = Path(filename).stem
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


def process_archive(archive_path: Path) -> list:
    """Process a single Takeout archive"""
    entries = []
    stats = defaultdict(int)

    print(f"[*] Processing: {archive_path.name}")

    try:
        if archive_path.suffix in ['.zip']:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for name in zf.namelist():
                    stats['total_files'] += 1

                    # Process .docx files
                    if name.endswith('.docx'):
                        # Check filename for AoR hints
                        if not is_aor_filename(name):
                            # Still check content for some files
                            if stats['docx_checked'] > 100:
                                continue

                        stats['docx_checked'] += 1

                        try:
                            docx_bytes = zf.read(name)
                            text = extract_docx_text(docx_bytes)

                            if not is_aor_content(text):
                                continue

                            stats['aor_found'] += 1

                            structure = parse_structure(text)
                            title = extract_title(text, name)

                            entries.append({
                                'id': f"takeout_{archive_path.stem}_{stats['aor_found']}",
                                'title': title,
                                'timestamp': '',  # Could extract from archive metadata
                                'source': 'google_takeout',
                                'source_file': name,
                                'archive': archive_path.name,
                                'structure': {
                                    'verse_count': len(structure['verses']),
                                    'has_hook': len(structure['hooks']) > 0,
                                    'has_bridge': len(structure['bridges']) > 0,
                                    'bar_count': count_bars(text),
                                },
                                'verses': structure['verses'],
                                'hooks': structure['hooks'],
                                'bridges': structure['bridges'],
                                'raw_text': text[:8000],
                            })

                            if stats['aor_found'] % 10 == 0:
                                print(f"    Found {stats['aor_found']} AoR tracks...")

                        except Exception as e:
                            stats['errors'] += 1

                    # Also check .json files for conversations
                    elif name.endswith('.json') and 'conversation' in name.lower():
                        try:
                            content = zf.read(name).decode('utf-8', errors='ignore')
                            if is_aor_content(content):
                                data = json.loads(content)
                                # Process as conversation JSON
                                stats['json_found'] += 1
                        except:
                            pass

        elif archive_path.suffix in ['.tgz', '.tar.gz']:
            with tarfile.open(archive_path, 'r:gz') as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    stats['total_files'] += 1

                    if member.name.endswith('.docx') and is_aor_filename(member.name):
                        stats['docx_checked'] += 1

                        try:
                            f = tf.extractfile(member)
                            if f:
                                docx_bytes = f.read()
                                text = extract_docx_text(docx_bytes)

                                if not is_aor_content(text):
                                    continue

                                stats['aor_found'] += 1
                                structure = parse_structure(text)
                                title = extract_title(text, member.name)

                                entries.append({
                                    'id': f"takeout_{archive_path.stem}_{stats['aor_found']}",
                                    'title': title,
                                    'timestamp': '',
                                    'source': 'google_takeout',
                                    'source_file': member.name,
                                    'archive': archive_path.name,
                                    'structure': {
                                        'verse_count': len(structure['verses']),
                                        'has_hook': len(structure['hooks']) > 0,
                                        'has_bridge': len(structure['bridges']) > 0,
                                        'bar_count': count_bars(text),
                                    },
                                    'verses': structure['verses'],
                                    'hooks': structure['hooks'],
                                    'bridges': structure['bridges'],
                                    'raw_text': text[:8000],
                                })
                        except Exception as e:
                            stats['errors'] += 1

    except Exception as e:
        print(f"    [!] Error: {e}")

    print(f"    → Found {stats['aor_found']} AoR tracks ({stats['docx_checked']} docs checked)")
    return entries


def save_catalog(entries: list):
    """Save extracted entries"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"aor_takeout_{timestamp}.jsonl"

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
        'source': 'google_takeout',
        'total_tracks': len(entries),
        'total_verses': total_verses,
        'total_bars': total_bars,
    }

    stats_path = OUTPUT_DIR / "takeout_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"[+] Stats: {stats_path}")
    print(f"\nNow run: python3 ~/aor-dmma/merge_catalogs.py")

    return output_path


def list_archives():
    """List all known takeout archives with estimated AoR content"""
    print("\n" + "="*70)
    print("KNOWN TAKEOUT ARCHIVES")
    print("="*70)

    takeout_dirs = [
        Path.home() / "Documents/OrganizedTakeout",
        Path.home() / "My Drive/All the filles/Takeout",
    ]

    archives = []
    for td in takeout_dirs:
        if td.exists():
            archives.extend(td.rglob("*.zip"))
            archives.extend(td.rglob("*.tgz"))

    # Dedupe by name
    seen = set()
    unique = []
    for a in archives:
        if a.name not in seen:
            seen.add(a.name)
            unique.append(a)

    print(f"\nFound {len(unique)} unique archives\n")

    # Sort by date in filename
    for a in sorted(unique, key=lambda x: x.name)[:20]:
        size_mb = a.stat().st_size / (1024*1024)
        priority = "★" if a in PRIORITY_ARCHIVES else " "
        print(f"  {priority} {a.name} ({size_mb:.1f} MB)")

    if len(unique) > 20:
        print(f"  ... and {len(unique) - 20} more")

    print("\n★ = Priority archive (high AoR content)")


def main():
    if '--list' in sys.argv:
        list_archives()
        return

    archives_to_process = []

    # Check for specific archive argument
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            continue
        p = Path(arg).expanduser().resolve()
        if p.exists():
            archives_to_process.append(p)

    # Default to priority archives
    if not archives_to_process:
        archives_to_process = [a for a in PRIORITY_ARCHIVES if a.exists()]

    if not archives_to_process:
        print("[!] No archives found. Use --list to see available archives.")
        return

    print(f"\n[*] Processing {len(archives_to_process)} archive(s)...\n")

    all_entries = []
    for archive in archives_to_process:
        entries = process_archive(archive)
        all_entries.extend(entries)

    if all_entries:
        save_catalog(all_entries)
    else:
        print("[!] No AoR content found")


if __name__ == "__main__":
    main()
