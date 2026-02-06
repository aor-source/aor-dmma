#!/usr/bin/env python3
"""
Gemini Chat Scout
=================
Scans ALL chats quickly, identifies research content by keywords.
Outputs a ranked list so you know which tabs to open.

Phase 1: Run this to find research chats
Phase 2: Open the flagged chats, run gemini_research_extractor.py

Usage:
    python3 gemini_scout.py
"""

import asyncio
import json
import re
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from playwright.async_api import async_playwright
except ImportError:
    os.system("pip install playwright --break-system-packages")
    from playwright.async_api import async_playwright


OUTPUT_DIR = Path.home() / "aor-dmma" / "scout_results"

# Research topic keywords - weighted by importance
RESEARCH_KEYWORDS = {
    # AI/ML theory
    "dead neuron": 10,
    "dying relu": 10,
    "neuron death": 10,
    "gradient": 5,
    "backprop": 5,
    "activation": 4,

    # Aesthetics
    "aesthetic": 10,
    "aesthetics": 10,
    "beauty": 5,
    "qualia": 10,
    "phenomenal": 7,
    "subjective": 5,

    # Emotional AI
    "emotion target": 10,
    "emotional target": 10,
    "affect": 6,
    "valence": 8,
    "arousal": 8,
    "sentiment": 5,
    "feeling": 4,

    # Emoji/language
    "emoji": 10,
    "emoticon": 6,
    "unicode": 5,
    "symbol": 4,
    "semantic": 6,
    "embedding": 5,

    # Deepfake
    "deepfake": 10,
    "fake detection": 10,
    "synthetic": 6,
    "generated": 4,
    "GAN": 7,
    "diffusion": 6,
    "forgery": 8,

    # Math/formal
    "equation": 6,
    "formula": 6,
    "theorem": 8,
    "proof": 7,
    "axiom": 8,
    "entropy": 7,
    "information theory": 9,
    "probability": 5,

    # Code/scripts
    "script": 4,
    "function": 3,
    "algorithm": 6,
    "implementation": 5,
    "python": 4,
    "code": 3,

    # Architecture
    "architecture": 6,
    "framework": 5,
    "model": 4,
    "transformer": 7,
    "attention": 6,

    # The Architect
    "architect": 8,
    "AoR": 8,
    "rhyme": 5,
}

# Categories for grouping
CATEGORIES = {
    "dead_neuron": ["dead neuron", "dying relu", "neuron death", "gradient", "activation"],
    "aesthetics": ["aesthetic", "aesthetics", "beauty", "qualia", "phenomenal"],
    "emotional_ai": ["emotion", "affect", "valence", "arousal", "sentiment", "feeling"],
    "emoji_language": ["emoji", "emoticon", "unicode", "symbol", "semantic"],
    "deepfake": ["deepfake", "fake detection", "synthetic", "GAN", "forgery"],
    "math_formal": ["equation", "formula", "theorem", "proof", "axiom", "entropy"],
    "code_scripts": ["script", "function", "algorithm", "implementation", "python", "code"],
    "architect_aor": ["architect", "AoR", "rhyme"],
}


def score_content(text):
    """Score content by research keyword matches"""
    text_lower = text.lower()
    score = 0
    matches = []
    categories_found = set()

    for keyword, weight in RESEARCH_KEYWORDS.items():
        count = len(re.findall(re.escape(keyword.lower()), text_lower))
        if count > 0:
            score += count * weight
            matches.append(f"{keyword} ({count}x)")

            # Track categories
            for cat, keywords in CATEGORIES.items():
                if any(k in keyword.lower() for k in keywords):
                    categories_found.add(cat)

    return score, matches, list(categories_found)


async def get_all_conversations(page):
    """Find all conversation links"""
    selectors = [
        'a[href*="/app/"]',
        '[data-conversation-id]',
        'div[role="listitem"] a',
        'aside a[href]',
        '[class*="conversation"] a',
        '[class*="history"] a',
    ]

    for selector in selectors:
        try:
            elements = await page.query_selector_all(selector)
            # Filter to actual chat links
            valid = []
            for el in elements:
                href = await el.get_attribute('href')
                if href and '/app/' in href:
                    valid.append(el)
            if len(valid) > 2:
                return valid
        except:
            continue

    return []


async def scroll_sidebar_fully(page):
    """Scroll sidebar to load ALL conversations"""
    print("📜 Loading all conversations from sidebar...")

    prev_count = 0
    stable = 0

    for attempt in range(50):  # Max 50 scroll attempts
        convs = await get_all_conversations(page)
        count = len(convs)

        if count == prev_count:
            stable += 1
            if stable >= 4:
                break
        else:
            stable = 0
            if count % 20 == 0:
                print(f"   Loaded {count} conversations...")

        prev_count = count

        # Scroll sidebar
        try:
            await page.evaluate('''
                const sidebar = document.querySelector('aside, nav, [class*="sidebar"]');
                if (sidebar) sidebar.scrollTop += 500;
            ''')
        except:
            pass

        await page.mouse.wheel(0, 300)
        await asyncio.sleep(0.5)

    print(f"   ✅ Found {prev_count} total conversations\n")
    return prev_count


async def quick_scan_chat(page, element, chat_num, total):
    """Quickly scan a single chat for research content"""
    result = {
        "number": chat_num,
        "title": "",
        "preview": "",
        "score": 0,
        "matches": [],
        "categories": [],
        "url": "",
    }

    try:
        # Click to open
        await element.click(timeout=5000)
        await asyncio.sleep(1.5)  # Brief load time

        result["url"] = page.url
        result["title"] = await page.title()

        # Get preview (first ~1000 chars of main content)
        try:
            main = await page.query_selector('main')
            if main:
                text = await main.inner_text()
                result["preview"] = text[:1500]
        except:
            pass

        # Score it
        combined = f"{result['title']} {result['preview']}"
        result["score"], result["matches"], result["categories"] = score_content(combined)

    except Exception as e:
        result["error"] = str(e)

    return result


async def run_scout():
    """Scout all chats and rank by research relevance"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "scan_started": datetime.now().isoformat(),
        "chats": [],
        "high_value": [],  # score > 50
        "medium_value": [],  # score 20-50
        "categories": defaultdict(list),
    }

    async with async_playwright() as p:
        print("🔗 Connecting to Chrome...")

        try:
            browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        except Exception as e:
            print(f"❌ Could not connect: {e}")
            print("\nStart Chrome with:")
            print('  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222')
            print("\nThen navigate to gemini.google.com")
            return

        # Find Gemini page
        gemini_page = None
        for ctx in browser.contexts:
            for page in ctx.pages:
                if 'gemini.google.com' in page.url:
                    gemini_page = page
                    break

        if not gemini_page:
            print("❌ No Gemini tab found. Open gemini.google.com first.")
            return

        print(f"✅ Connected to Gemini\n")

        # Load all conversations
        total = await scroll_sidebar_fully(gemini_page)
        conversations = await get_all_conversations(gemini_page)

        print(f"🔍 Scanning {len(conversations)} conversations for research content...\n")

        for i, conv in enumerate(conversations):
            chat_num = i + 1

            if chat_num % 10 == 0:
                print(f"   Scanning {chat_num}/{len(conversations)}...")

            result = await quick_scan_chat(gemini_page, conv, chat_num, len(conversations))
            results["chats"].append(result)

            # Categorize by score
            if result["score"] >= 50:
                results["high_value"].append(result)
                print(f"   🎯 HIGH VALUE: {result['title'][:50]} (score: {result['score']})")
            elif result["score"] >= 20:
                results["medium_value"].append(result)

            # Track by category
            for cat in result["categories"]:
                results["categories"][cat].append(result["title"])

            # Small delay between chats
            await asyncio.sleep(0.3)

        # Sort by score
        results["chats"].sort(key=lambda x: x["score"], reverse=True)
        results["high_value"].sort(key=lambda x: x["score"], reverse=True)
        results["medium_value"].sort(key=lambda x: x["score"], reverse=True)

        results["scan_completed"] = datetime.now().isoformat()
        results["total_scanned"] = len(conversations)
        results["high_value_count"] = len(results["high_value"])
        results["medium_value_count"] = len(results["medium_value"])

        # Save full results
        with open(OUTPUT_DIR / "scout_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=list)

        # Save human-readable summary
        summary_path = OUTPUT_DIR / "RESEARCH_CHATS.txt"
        with open(summary_path, 'w') as f:
            f.write("GEMINI RESEARCH CHAT SCOUT RESULTS\n")
            f.write(f"{'='*60}\n")
            f.write(f"Scanned: {results['total_scanned']} chats\n")
            f.write(f"High value (score 50+): {results['high_value_count']}\n")
            f.write(f"Medium value (score 20-50): {results['medium_value_count']}\n")
            f.write(f"\n{'='*60}\n")
            f.write("HIGH VALUE RESEARCH CHATS - OPEN THESE:\n")
            f.write(f"{'='*60}\n\n")

            for chat in results["high_value"]:
                f.write(f"📌 [{chat['score']}] {chat['title']}\n")
                f.write(f"   Categories: {', '.join(chat['categories'])}\n")
                f.write(f"   Matches: {', '.join(chat['matches'][:5])}\n")
                f.write(f"   URL: {chat['url']}\n\n")

            f.write(f"\n{'='*60}\n")
            f.write("MEDIUM VALUE CHATS:\n")
            f.write(f"{'='*60}\n\n")

            for chat in results["medium_value"][:20]:
                f.write(f"📎 [{chat['score']}] {chat['title']}\n")
                f.write(f"   Categories: {', '.join(chat['categories'])}\n\n")

            f.write(f"\n{'='*60}\n")
            f.write("BY CATEGORY:\n")
            f.write(f"{'='*60}\n\n")

            for cat, titles in results["categories"].items():
                f.write(f"\n{cat.upper()} ({len(titles)} chats):\n")
                for title in titles[:10]:
                    f.write(f"  • {title[:60]}\n")

        print(f"\n{'='*60}")
        print(f"🎯 SCOUT COMPLETE")
        print(f"{'='*60}")
        print(f"   Total scanned: {results['total_scanned']}")
        print(f"   HIGH VALUE:    {results['high_value_count']} chats")
        print(f"   Medium value:  {results['medium_value_count']} chats")
        print(f"\n   Results: {OUTPUT_DIR / 'RESEARCH_CHATS.txt'}")
        print(f"{'='*60}\n")

        print("\nNEXT STEPS:")
        print("1. Open RESEARCH_CHATS.txt to see which chats to extract")
        print("2. Open those specific tabs in Chrome")
        print("3. Run: python3 ~/aor-dmma/gemini_research_extractor.py")

    return results


if __name__ == "__main__":
    asyncio.run(run_scout())
