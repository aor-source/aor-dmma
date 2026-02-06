#!/usr/bin/env python3
"""
Gemini Conversation Extractor v2
================================
Fixed version with updated selectors and wait strategies.

Changes from v1:
- Removed networkidle wait (Gemini keeps connections open)
- Updated selectors for 2026 Gemini UI
- Better content extraction
- More robust scrolling

Usage:
    python3 gemini_extractor_v2.py
    python3 gemini_extractor_v2.py --output-dir ~/Desktop/gemini_corpus
"""

import asyncio
import json
import random
import re
import os
from pathlib import Path
from datetime import datetime

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    os.system("pip install playwright --break-system-packages")
    os.system("playwright install chromium")
    from playwright.async_api import async_playwright


OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", os.path.expanduser("~/aor-dmma/aor_catalog/gemini_corpus")))
SESSION_FILE = Path.home() / ".config" / "gemini_session.json"

# AoR detection
AOR_PATTERNS = [
    r'\[Verse\s*\d*\]',
    r'\[Hook\]',
    r'\[Chorus\]',
    r'\[Bridge\]',
    r'Architect of Rhyme',
    r'— AoR',
    r'I am the Architect',
]


def is_aor_content(text):
    return any(re.search(p, text, re.IGNORECASE) for p in AOR_PATTERNS)


def human_delay(min_s=0.5, max_s=1.5):
    base = random.uniform(min_s, max_s)
    if random.random() < 0.1:
        base += random.uniform(0.5, 1.5)
    return base


async def safe_click(element):
    """Click with retry"""
    try:
        await element.click(timeout=5000)
        return True
    except:
        try:
            await element.click(force=True, timeout=5000)
            return True
        except:
            return False


async def get_all_conversations(page):
    """Find all conversation links in sidebar"""
    await asyncio.sleep(2)

    # Try multiple selector patterns for Gemini sidebar
    selectors = [
        'a[href*="/app/"]',
        '[data-conversation-id]',
        'div[role="listitem"] a',
        'nav a[href*="conversation"]',
        'aside a',
        '[class*="conversation"] a',
        '[class*="chat-list"] a',
        '[class*="history"] a',
    ]

    conversations = []
    for selector in selectors:
        try:
            elements = await page.query_selector_all(selector)
            if len(elements) > 2:
                print(f"    Found {len(elements)} conversations with: {selector}")
                conversations = elements
                break
        except:
            continue

    return conversations


async def scroll_sidebar(page):
    """Scroll sidebar to load all conversations"""
    print("\n📜 Loading all conversations...")

    # Find sidebar
    sidebar = None
    sidebar_selectors = [
        'aside',
        'nav',
        '[class*="sidebar"]',
        '[class*="drawer"]',
        '[role="navigation"]',
    ]

    for sel in sidebar_selectors:
        try:
            elem = await page.query_selector(sel)
            if elem:
                box = await elem.bounding_box()
                if box and box['width'] < 400:  # Sidebar is narrow
                    sidebar = elem
                    break
        except:
            continue

    prev_count = 0
    stable = 0

    for _ in range(30):  # Max 30 scroll attempts
        convs = await get_all_conversations(page)
        count = len(convs)

        if count == prev_count:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
            print(f"    Loaded {count} conversations...")

        prev_count = count

        # Scroll
        try:
            if sidebar:
                await sidebar.evaluate('el => el.scrollTop += 500')
            else:
                await page.mouse.wheel(0, 300)
        except:
            await page.mouse.wheel(0, 300)

        await asyncio.sleep(human_delay(0.8, 1.5))

    print(f"    ✅ Total: {prev_count} conversations\n")
    return prev_count


async def extract_current_chat(page):
    """Extract content from currently visible chat"""
    content = {
        "messages": [],
        "raw_text": "",
        "extracted_at": datetime.now().isoformat(),
        "is_aor": False,
    }

    await asyncio.sleep(human_delay(1.5, 3.0))

    # Try multiple content selectors
    message_selectors = [
        'div[class*="response"]',
        'div[class*="message"]',
        'div[class*="turn"]',
        'div[class*="model"]',
        'article',
        'main div[class*="content"]',
        '[data-message-author-role]',
    ]

    messages = []
    for selector in message_selectors:
        try:
            elements = await page.query_selector_all(selector)
            if len(elements) >= 2:
                for el in elements:
                    text = await el.inner_text()
                    text = text.strip()
                    if text and len(text) > 20:
                        messages.append(text)
                if messages:
                    break
        except:
            continue

    # Fallback: get main content area
    if not messages:
        try:
            main = await page.query_selector('main')
            if main:
                text = await main.inner_text()
                if text:
                    messages = [text]
        except:
            pass

    # Scroll to load more content
    for _ in range(5):
        try:
            await page.mouse.wheel(0, 500)
            await asyncio.sleep(0.5)
        except:
            break

    # Get any additional content after scroll
    try:
        main = await page.query_selector('main')
        if main:
            full_text = await main.inner_text()
            if full_text and len(full_text) > len('\n'.join(messages)):
                messages = [full_text]
    except:
        pass

    content["messages"] = messages
    content["raw_text"] = "\n\n---\n\n".join(messages)
    content["message_count"] = len(messages)
    content["is_aor"] = is_aor_content(content["raw_text"])

    return content


async def get_chat_title(page):
    """Get conversation title"""
    selectors = [
        'h1',
        '[class*="title"]',
        'header span',
        'main h1',
        'main h2',
    ]

    for selector in selectors:
        try:
            elements = await page.query_selector_all(selector)
            for el in elements:
                text = await el.inner_text()
                text = text.strip()
                if text and 3 < len(text) < 200:
                    return text
        except:
            continue

    return f"chat_{datetime.now().strftime('%H%M%S')}"


def sanitize_filename(name, max_len=60):
    clean = re.sub(r'[^\w\s-]', '', name)
    clean = re.sub(r'\s+', '_', clean.strip())
    return clean[:max_len] if clean else "untitled"


async def run_extraction(start_from=0, use_existing_chrome=False, cdp_url=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "extraction_started": datetime.now().isoformat(),
        "conversations": [],
        "aor_tracks": [],
        "total_extracted": 0,
        "total_aor": 0,
    }

    async with async_playwright() as p:
        browser = None
        context = None

        # Option 1: Connect to existing Chrome via CDP
        if use_existing_chrome or cdp_url:
            url = cdp_url or "http://127.0.0.1:9222"
            print(f"🔗 Connecting to existing Chrome at {url}")
            print("   (Start Chrome with: /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222)")
            try:
                browser = await p.chromium.connect_over_cdp(url)
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = context.pages[0] if context.pages else await context.new_page()
                print("   ✅ Connected to existing Chrome session!")
            except Exception as e:
                print(f"   ❌ Could not connect: {e}")
                print("   Falling back to new browser with saved session...")
                browser = None

        # Option 2: New browser with saved session state
        if not browser:
            print("🌐 Launching browser...")

            browser = await p.chromium.launch(
                headless=False,
                slow_mo=30,
                args=['--disable-blink-features=AutomationControlled']
            )

            # Load saved session if exists
            if SESSION_FILE.exists():
                print(f"   Loading saved session from {SESSION_FILE}")
                try:
                    context = await browser.new_context(
                        storage_state=str(SESSION_FILE),
                        viewport={"width": 1400, "height": 900},
                        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                    )
                    print("   ✅ Session loaded!")
                except Exception as e:
                    print(f"   ⚠️ Could not load session: {e}")
                    context = None

            if not context:
                context = await browser.new_context(
                    viewport={"width": 1400, "height": 900},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                )

            page = await context.new_page()

        # Go to Gemini
        await page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
        await asyncio.sleep(3)

        # Check if already logged in
        logged_in = False
        try:
            # Look for signs of being logged in (sidebar with chats)
            convs = await get_all_conversations(page)
            if len(convs) > 0:
                logged_in = True
                print("   ✅ Already logged in!")
        except:
            pass

        if not logged_in:
            input("\n✋ Log in to Google, then press ENTER when you see your chats... ")
            # Save session for next time
            await context.storage_state(path=str(SESSION_FILE))
            print(f"   💾 Session saved to {SESSION_FILE}")

        print("\n🚀 Starting extraction...\n")
        await asyncio.sleep(2)

        # Load all conversations
        total = await scroll_sidebar(page)

        # Get conversation links
        conversations = await get_all_conversations(page)
        print(f"📋 Found {len(conversations)} clickable conversations\n")

        if start_from > 0:
            print(f"   Skipping first {start_from}")
            conversations = conversations[start_from:]

        for i, conv in enumerate(conversations):
            chat_num = i + start_from + 1
            print(f"\n{'='*50}")
            print(f"📖 Chat {chat_num}/{total}")

            try:
                # Click the conversation
                clicked = await safe_click(conv)
                if not clicked:
                    print("   ⚠️ Couldn't click, skipping")
                    continue

                # Wait for content to load (fixed delay, not networkidle)
                await asyncio.sleep(human_delay(2.0, 4.0))

                # Get title
                title = await get_chat_title(page)
                print(f"   📝 {title[:50]}")

                # Extract content
                content = await extract_current_chat(page)
                content["title"] = title
                content["chat_number"] = chat_num

                # Save
                filename = f"{chat_num:04d}_{sanitize_filename(title)}.json"
                filepath = OUTPUT_DIR / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)

                # Also save as txt
                txt_path = OUTPUT_DIR / f"{chat_num:04d}_{sanitize_filename(title)}.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"TITLE: {title}\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(content['raw_text'])

                manifest["conversations"].append({
                    "number": chat_num,
                    "title": title,
                    "filename": filename,
                    "is_aor": content["is_aor"],
                    "text_length": len(content["raw_text"]),
                })
                manifest["total_extracted"] += 1

                if content["is_aor"]:
                    manifest["aor_tracks"].append(filename)
                    manifest["total_aor"] += 1
                    print(f"   🎤 AoR CONTENT FOUND!")

                print(f"   ✅ Saved ({len(content['raw_text'])} chars)")

                # Save manifest periodically
                with open(OUTPUT_DIR / "manifest.json", 'w') as f:
                    json.dump(manifest, f, indent=2)

                # Occasional longer pause
                if random.random() < 0.1:
                    pause = random.uniform(3, 8)
                    print(f"   ☕ Brief pause ({pause:.1f}s)")
                    await asyncio.sleep(pause)

            except Exception as e:
                print(f"   ❌ Error: {e}")
                manifest["conversations"].append({
                    "number": chat_num,
                    "error": str(e),
                })

        # Final save
        manifest["extraction_completed"] = datetime.now().isoformat()
        with open(OUTPUT_DIR / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{'='*50}")
        print(f"🎉 EXTRACTION COMPLETE")
        print(f"{'='*50}")
        print(f"   Total conversations: {manifest['total_extracted']}")
        print(f"   AoR tracks found: {manifest['total_aor']}")
        print(f"   Output: {OUTPUT_DIR}")
        print(f"{'='*50}\n")

        await browser.close()

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Gemini conversations")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N chats")
    parser.add_argument("--use-chrome", action="store_true", help="Connect to existing Chrome (must be started with --remote-debugging-port=9222)")
    parser.add_argument("--cdp-url", default=None, help="Chrome DevTools Protocol URL (default: http://127.0.0.1:9222)")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    asyncio.run(run_extraction(args.start_from, args.use_chrome, args.cdp_url))
