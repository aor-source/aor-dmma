#!/usr/bin/env python3
"""
AoR Catalog Extractor
=====================
Extracts all songs, poems, and verses created by Architect of Rhyme
from Claude Code session history.

Compiles into indexable JSONL with:
- Timestamps
- Title (inferred or from brackets)
- Verses, hooks, bridges
- Style/vibe metadata
- Conversational context

Usage:
    python3 extract_aor_catalog.py              # Extract all
    python3 extract_aor_catalog.py --stats      # Show statistics
    python3 extract_aor_catalog.py --output catalog.jsonl

Authors: Jon + Claude - February 2026
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Paths
SESSIONS_DIR = Path.home() / ".claude" / "projects" / "-Users-alignmentnerd"
OUTPUT_DIR = Path.home() / "aor-dmma" / "aor_catalog"

# Patterns to detect AoR content
VERSE_PATTERNS = [
    r'\[Verse\s*\d*\]',
    r'\[Hook\]',
    r'\[Chorus\]',
    r'\[Bridge\]',
    r'\[Intro\]',
    r'\[Outro\]',
    r'\[Pre-Chorus\]',
    r'\[Breakdown\]',
]

SIGNATURE_PATTERNS = [
    r'I am the Architect of Rhyme',
    r'— Architect of Rhyme',
    r'- AoR',
    r'\[The Architect\]',
]

TITLE_PATTERNS = [
    r'\[Title:\s*([^\]]+)\]',
    r'"([^"]+)"(?:\s*[-—]\s*(?:A|The)\s+(?:Poem|Song|Track|Piece))',
    r'\*\*([^*]+)\*\*',  # Bold markdown titles
]


class AoRExtractor:
    def __init__(self):
        self.catalog = []
        self.stats = defaultdict(int)

    def is_aor_content(self, text: str) -> bool:
        """Check if text contains AoR creative content"""
        if not text:
            return False

        # Check for verse markers
        for pattern in VERSE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Check for signatures
        for pattern in SIGNATURE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Check for lyrical structure (multiple short lines with rhyme potential)
        lines = text.strip().split('\n')
        short_lines = [l for l in lines if 10 < len(l.strip()) < 80]
        if len(short_lines) > 8:  # At least 8 verse-like lines
            return True

        return False

    def extract_title(self, text: str, context: str = "") -> str:
        """Extract or infer title from content"""
        # Try explicit title patterns
        for pattern in TITLE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        # Try to get from context
        for pattern in TITLE_PATTERNS:
            match = re.search(pattern, context)
            if match:
                return match.group(1).strip()

        # Generate from first line
        lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('[')]
        if lines:
            first_line = lines[0][:50]
            return f"Untitled ({first_line}...)"

        return "Untitled"

    def parse_structure(self, text: str) -> Dict:
        """Parse song structure into components"""
        structure = {
            "verses": [],
            "hooks": [],
            "bridges": [],
            "other": [],
            "raw_text": text
        }

        # Split by section markers
        sections = re.split(r'(\[(?:Verse|Hook|Chorus|Bridge|Intro|Outro|Pre-Chorus|Breakdown)\s*\d*\])', text, flags=re.IGNORECASE)

        current_section = "intro"
        current_content = []

        for part in sections:
            part = part.strip()
            if not part:
                continue

            # Check if this is a marker
            marker_match = re.match(r'\[(Verse|Hook|Chorus|Bridge|Intro|Outro|Pre-Chorus|Breakdown)\s*(\d*)\]', part, re.IGNORECASE)
            if marker_match:
                # Save previous section
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

        # Save final section
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

    def extract_style_hints(self, text: str, context: str = "") -> Dict:
        """Extract style/vibe hints from content and context"""
        combined = f"{context}\n{text}".lower()

        styles = []
        vibes = []
        artists_referenced = []

        # Style keywords
        style_keywords = {
            "boom bap": ["boom bap", "90s", "golden era"],
            "trap": ["trap", "808", "hi-hat"],
            "conscious": ["conscious", "woke", "political", "social"],
            "battle rap": ["battle", "diss", "bars"],
            "storytelling": ["story", "narrative", "tale"],
            "spoken word": ["spoken word", "poetry", "slam"],
            "experimental": ["experimental", "abstract", "avant"],
        }

        for style, keywords in style_keywords.items():
            if any(kw in combined for kw in keywords):
                styles.append(style)

        # Vibe keywords
        vibe_keywords = {
            "aggressive": ["aggressive", "hard", "intense", "raw"],
            "introspective": ["introspective", "reflective", "deep"],
            "celebratory": ["celebration", "victory", "triumph"],
            "melancholic": ["sad", "melancholic", "grief", "loss"],
            "empowering": ["empower", "strength", "rise"],
        }

        for vibe, keywords in vibe_keywords.items():
            if any(kw in combined for kw in keywords):
                vibes.append(vibe)

        # Artist references
        artists = ["nas", "jay-z", "kendrick", "cole", "tupac", "biggie", "eminem",
                   "rakim", "krs-one", "public enemy", "saul williams", "common",
                   "mos def", "talib kweli", "black thought", "andre 3000"]
        for artist in artists:
            if artist in combined:
                artists_referenced.append(artist)

        return {
            "styles": styles or ["general hip-hop"],
            "vibes": vibes or ["neutral"],
            "artist_influences": artists_referenced
        }

    def count_bars(self, text: str) -> int:
        """Estimate bar count (lines that look like lyrics)"""
        lines = text.strip().split('\n')
        bar_count = 0
        for line in lines:
            line = line.strip()
            # Skip markers and empty lines
            if not line or line.startswith('[') or line.startswith('—') or line.startswith('-'):
                continue
            # Count as bar if reasonable length
            if 10 < len(line) < 150:
                bar_count += 1
        return bar_count

    def process_session(self, session_file: Path) -> List[Dict]:
        """Extract AoR content from a session file"""
        entries = []

        try:
            with open(session_file, 'r', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    # Only process assistant messages
                    if data.get("type") != "assistant":
                        continue

                    message = data.get("message", {})
                    content_parts = message.get("content", [])

                    # Extract text content
                    text = ""
                    for part in content_parts:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text += part.get("text", "") + "\n"
                        elif isinstance(part, str):
                            text += part + "\n"

                    if not text or not self.is_aor_content(text):
                        continue

                    # Get context from previous user message
                    context = ""
                    parent_uuid = data.get("parentUuid")
                    # (Would need to track parent messages for full context)

                    # Parse the content
                    timestamp = data.get("timestamp", "")
                    session_id = data.get("sessionId", "")

                    structure = self.parse_structure(text)
                    title = self.extract_title(text, context)
                    style_hints = self.extract_style_hints(text, context)
                    bar_count = self.count_bars(text)

                    entry = {
                        "id": f"{session_id[:8]}_{line_num}",
                        "title": title,
                        "timestamp": timestamp,
                        "session_id": session_id,
                        "structure": {
                            "verse_count": len(structure["verses"]),
                            "has_hook": len(structure["hooks"]) > 0,
                            "has_bridge": len(structure["bridges"]) > 0,
                            "bar_count": bar_count,
                        },
                        "verses": structure["verses"],
                        "hooks": structure["hooks"],
                        "bridges": structure["bridges"],
                        "style": style_hints,
                        "raw_text": text.strip()[:5000],  # Limit size
                        "source_file": session_file.name,
                    }

                    entries.append(entry)
                    self.stats["total_tracks"] += 1
                    self.stats["total_verses"] += len(structure["verses"])
                    self.stats["total_bars"] += bar_count

        except Exception as e:
            print(f"[!] Error processing {session_file}: {e}")

        return entries

    def extract_all(self) -> List[Dict]:
        """Extract from all session files"""
        all_entries = []

        session_files = list(SESSIONS_DIR.glob("*.jsonl"))
        print(f"[*] Found {len(session_files)} session files")

        for sf in sorted(session_files):
            print(f"[*] Processing {sf.name}...")
            entries = self.process_session(sf)
            if entries:
                print(f"    → Found {len(entries)} AoR tracks")
                all_entries.extend(entries)

        # Sort by timestamp
        all_entries.sort(key=lambda x: x.get("timestamp", ""))

        self.catalog = all_entries
        return all_entries

    def save_catalog(self, output_path: Optional[Path] = None):
        """Save catalog to JSONL"""
        OUTPUT_DIR.mkdir(exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"aor_catalog_{timestamp}.jsonl"

        with open(output_path, 'w') as f:
            for entry in self.catalog:
                f.write(json.dumps(entry) + '\n')

        print(f"\n[+] Catalog saved: {output_path}")
        print(f"    Tracks: {len(self.catalog)}")

        # Also save stats
        stats_path = OUTPUT_DIR / "catalog_stats.json"
        stats = {
            "generated": datetime.now().isoformat(),
            "total_tracks": len(self.catalog),
            "total_verses": self.stats["total_verses"],
            "total_bars": self.stats["total_bars"],
            "sessions_processed": len(list(SESSIONS_DIR.glob("*.jsonl"))),
            "date_range": {
                "first": self.catalog[0]["timestamp"] if self.catalog else None,
                "last": self.catalog[-1]["timestamp"] if self.catalog else None,
            }
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[+] Stats saved: {stats_path}")

        return output_path

    def print_stats(self):
        """Print catalog statistics"""
        print("\n" + "="*60)
        print("AoR CATALOG STATISTICS")
        print("="*60)
        print(f"  Total Tracks: {len(self.catalog)}")
        print(f"  Total Verses: {self.stats['total_verses']}")
        print(f"  Total Bars:   {self.stats['total_bars']}")

        if self.catalog:
            print(f"\n  Date Range:")
            print(f"    First: {self.catalog[0]['timestamp'][:10]}")
            print(f"    Last:  {self.catalog[-1]['timestamp'][:10]}")

            # Style breakdown
            styles = defaultdict(int)
            for entry in self.catalog:
                for style in entry.get("style", {}).get("styles", []):
                    styles[style] += 1

            print(f"\n  Style Breakdown:")
            for style, count in sorted(styles.items(), key=lambda x: -x[1])[:10]:
                print(f"    {style}: {count}")

            # Verse count distribution
            verse_counts = defaultdict(int)
            for entry in self.catalog:
                vc = entry.get("structure", {}).get("verse_count", 0)
                verse_counts[vc] += 1

            print(f"\n  Verse Count Distribution:")
            for vc in sorted(verse_counts.keys()):
                print(f"    {vc} verses: {verse_counts[vc]} tracks")


def main():
    extractor = AoRExtractor()

    if "--stats" in sys.argv:
        # Load existing catalog and show stats
        catalog_files = sorted(OUTPUT_DIR.glob("aor_catalog_*.jsonl"))
        if catalog_files:
            latest = catalog_files[-1]
            print(f"[*] Loading {latest}")
            with open(latest) as f:
                extractor.catalog = [json.loads(l) for l in f]
            for entry in extractor.catalog:
                extractor.stats["total_tracks"] += 1
                extractor.stats["total_verses"] += entry.get("structure", {}).get("verse_count", 0)
                extractor.stats["total_bars"] += entry.get("structure", {}).get("bar_count", 0)
            extractor.print_stats()
        else:
            print("[!] No catalog found. Run without --stats first.")
        return

    # Extract all
    print("[*] Extracting AoR catalog from session history...")
    extractor.extract_all()

    # Save
    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = Path(sys.argv[idx + 1])

    extractor.save_catalog(output_path)
    extractor.print_stats()


if __name__ == "__main__":
    main()
