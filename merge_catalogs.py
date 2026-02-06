#!/usr/bin/env python3
"""
AoR Catalog Merger
==================
Merges Chrome-extracted JSONL with CLI-extracted catalog.
Deduplicates by content hash and normalizes format.

Usage:
    python3 merge_catalogs.py                    # Auto-find and merge
    python3 merge_catalogs.py chrome_export.jsonl
    python3 merge_catalogs.py --stats

Authors: Jon + Claude - February 2026
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

CATALOG_DIR = Path.home() / "aor-dmma" / "aor_catalog"


def content_hash(text: str) -> str:
    """Generate hash of normalized content for deduplication"""
    # Normalize: lowercase, remove extra whitespace
    normalized = ' '.join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def load_jsonl(filepath: Path) -> list:
    """Load JSONL file"""
    entries = []
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def normalize_entry(entry: dict, source: str) -> dict:
    """Normalize entry format from different sources"""
    # Handle Chrome format vs CLI format
    raw_text = entry.get('raw_text', '')

    normalized = {
        'id': entry.get('id', content_hash(raw_text)),
        'title': entry.get('title', 'Untitled'),
        'timestamp': entry.get('timestamp', entry.get('created_at', '')),
        'source': source,
        'structure': entry.get('structure', {
            'verse_count': len(entry.get('verses', [])),
            'has_hook': len(entry.get('hooks', [])) > 0,
            'has_bridge': len(entry.get('bridges', [])) > 0,
            'bar_count': entry.get('bar_count', 0),
        }),
        'verses': entry.get('verses', []),
        'hooks': entry.get('hooks', []),
        'bridges': entry.get('bridges', []),
        'style': entry.get('style', {}),
        'raw_text': raw_text,
        'content_hash': content_hash(raw_text),
    }

    # Add Chrome-specific fields if present
    if 'conversation_id' in entry:
        normalized['conversation_id'] = entry['conversation_id']
        normalized['conversation_name'] = entry.get('conversation_name', '')

    # Add CLI-specific fields if present
    if 'session_id' in entry:
        normalized['session_id'] = entry['session_id']
        normalized['source_file'] = entry.get('source_file', '')

    return normalized


def merge_catalogs(chrome_files: list = None, cli_files: list = None) -> list:
    """Merge catalogs from multiple sources, deduplicate"""

    all_entries = []
    seen_hashes = set()
    stats = defaultdict(int)

    # Find CLI catalogs
    if cli_files is None:
        cli_files = sorted(CATALOG_DIR.glob("aor_catalog_*.jsonl"))

    # Find Chrome exports
    if chrome_files is None:
        chrome_files = sorted(CATALOG_DIR.glob("aor_conversations*.jsonl"))
        # Also check Downloads
        downloads = Path.home() / "Downloads"
        chrome_files.extend(sorted(downloads.glob("aor_conversations*.jsonl")))

    print(f"[*] Found {len(cli_files)} CLI catalog(s)")
    print(f"[*] Found {len(chrome_files)} Chrome export(s)")

    # Process CLI catalogs
    for cf in cli_files:
        print(f"    Loading {cf.name}...")
        entries = load_jsonl(cf)
        for entry in entries:
            normalized = normalize_entry(entry, 'cli')
            if normalized['content_hash'] not in seen_hashes:
                seen_hashes.add(normalized['content_hash'])
                all_entries.append(normalized)
                stats['cli'] += 1
            else:
                stats['cli_dupes'] += 1

    # Process Chrome exports
    for cf in chrome_files:
        print(f"    Loading {cf.name}...")
        entries = load_jsonl(cf)
        for entry in entries:
            normalized = normalize_entry(entry, 'chrome')
            if normalized['content_hash'] not in seen_hashes:
                seen_hashes.add(normalized['content_hash'])
                all_entries.append(normalized)
                stats['chrome'] += 1
            else:
                stats['chrome_dupes'] += 1

    # Sort by timestamp
    all_entries.sort(key=lambda x: x.get('timestamp', '') or '')

    print(f"\n[+] Merge complete:")
    print(f"    CLI tracks:    {stats['cli']} (skipped {stats['cli_dupes']} dupes)")
    print(f"    Chrome tracks: {stats['chrome']} (skipped {stats['chrome_dupes']} dupes)")
    print(f"    Total unique:  {len(all_entries)}")

    return all_entries, stats


def save_merged(entries: list, stats: dict):
    """Save merged catalog"""
    CATALOG_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = CATALOG_DIR / f"aor_merged_{timestamp}.jsonl"

    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n[+] Saved: {output_path}")

    # Update stats
    total_bars = sum(e.get('structure', {}).get('bar_count', 0) for e in entries)
    total_verses = sum(len(e.get('verses', [])) for e in entries)

    stats_data = {
        'generated': datetime.now().isoformat(),
        'total_tracks': len(entries),
        'total_verses': total_verses,
        'total_bars': total_bars,
        'sources': {
            'cli': stats.get('cli', 0),
            'chrome': stats.get('chrome', 0),
        },
        'date_range': {
            'first': entries[0]['timestamp'] if entries else None,
            'last': entries[-1]['timestamp'] if entries else None,
        }
    }

    stats_path = CATALOG_DIR / "merged_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)

    print(f"[+] Stats: {stats_path}")

    return output_path


def print_stats():
    """Print stats from most recent merge"""
    stats_path = CATALOG_DIR / "merged_stats.json"
    if not stats_path.exists():
        stats_path = CATALOG_DIR / "catalog_stats.json"

    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        print("\n" + "="*60)
        print("AoR MERGED CATALOG STATISTICS")
        print("="*60)
        print(f"  Total Tracks: {stats.get('total_tracks', 0)}")
        print(f"  Total Verses: {stats.get('total_verses', 0)}")
        print(f"  Total Bars:   {stats.get('total_bars', 0)}")

        sources = stats.get('sources', {})
        if sources:
            print(f"\n  Sources:")
            print(f"    CLI:    {sources.get('cli', 0)}")
            print(f"    Chrome: {sources.get('chrome', 0)}")

        date_range = stats.get('date_range', {})
        if date_range.get('first'):
            print(f"\n  Date Range:")
            print(f"    First: {date_range['first'][:10]}")
            print(f"    Last:  {date_range['last'][:10]}")
    else:
        print("[!] No stats file found. Run merge first.")


def main():
    if '--stats' in sys.argv:
        print_stats()
        return

    # Check for specific file argument
    chrome_files = []
    for arg in sys.argv[1:]:
        if arg.endswith('.jsonl') and Path(arg).exists():
            chrome_files.append(Path(arg))

    entries, stats = merge_catalogs(
        chrome_files=chrome_files if chrome_files else None
    )

    if entries:
        save_merged(entries, stats)
    else:
        print("[!] No entries to merge")


if __name__ == "__main__":
    main()
