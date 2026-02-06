#!/usr/bin/env python3
"""
Gemini Research Corpus Extractor
=================================
Extracts EVERYTHING from open Gemini tabs.
No filtering - captures full research conversations.

For: AI aesthetics, dead neurons, emotional targets,
     emoji language, deepfake detection, scripts, etc.

Usage:
    1. Start Chrome with debugging:
       /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

    2. Open all your research chat tabs, scroll to load history

    3. Run:
       python3 gemini_research_extractor.py
"""

import asyncio
import json
import re
import os
from pathlib import Path
from datetime import datetime

try:
    from playwright.async_api import async_playwright
except ImportError:
    os.system("pip install playwright --break-system-packages")
    from playwright.async_api import async_playwright


OUTPUT_DIR = Path.home() / "aor-dmma" / "research_corpus"


async def extract_page_content(page):
    """Extract ALL content from a page"""
    content = {
        "url": page.url,
        "title": await page.title(),
        "extracted_at": datetime.now().isoformat(),
        "messages": [],
        "raw_text": "",
        "code_blocks": [],
        "math_blocks": [],
    }

    try:
        # Get full page text
        body = await page.query_selector('body')
        if body:
            content["raw_text"] = await body.inner_text()

        # Try to get structured messages
        message_selectors = [
            '[data-message-author-role]',
            'div[class*="message"]',
            'div[class*="response"]',
            'div[class*="turn"]',
            'div[class*="query"]',
            'article',
        ]

        for selector in message_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if len(elements) >= 2:
                    for el in elements:
                        text = await el.inner_text()
                        if text and len(text.strip()) > 10:
                            content["messages"].append(text.strip())
                    break
            except:
                continue

        # Extract code blocks
        code_selectors = ['pre', 'code', '[class*="code"]', '[class*="hljs"]']
        for selector in code_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for el in elements:
                    code = await el.inner_text()
                    if code and len(code.strip()) > 20:
                        content["code_blocks"].append(code.strip())
            except:
                continue

        # Extract math (LaTeX, equations)
        math_selectors = ['[class*="math"]', '[class*="katex"]', '[class*="latex"]', 'mjx-container']
        for selector in math_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for el in elements:
                    math = await el.inner_text()
                    if math:
                        content["math_blocks"].append(math.strip())
            except:
                continue

    except Exception as e:
        content["error"] = str(e)

    content["message_count"] = len(content["messages"])
    content["code_block_count"] = len(content["code_blocks"])
    content["math_block_count"] = len(content["math_blocks"])
    content["total_chars"] = len(content["raw_text"])

    return content


def sanitize_filename(name, max_len=80):
    clean = re.sub(r'[^\w\s-]', '', name)
    clean = re.sub(r'\s+', '_', clean.strip())
    return clean[:max_len] if clean else "untitled"


async def extract_all_tabs():
    """Extract content from all open Gemini tabs"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "extraction_started": datetime.now().isoformat(),
        "pages": [],
        "total_pages": 0,
        "total_chars": 0,
        "total_code_blocks": 0,
        "total_math_blocks": 0,
    }

    async with async_playwright() as p:
        print("🔗 Connecting to Chrome...")
        print("   Make sure Chrome is running with: --remote-debugging-port=9222\n")

        try:
            browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        except Exception as e:
            print(f"❌ Could not connect to Chrome: {e}")
            print("\nStart Chrome with:")
            print('  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222')
            return

        print(f"✅ Connected! Found {len(browser.contexts)} context(s)\n")

        all_pages = []
        for ctx in browser.contexts:
            all_pages.extend(ctx.pages)

        # Filter to Gemini pages
        gemini_pages = [p for p in all_pages if 'gemini.google.com' in p.url]
        print(f"📑 Found {len(gemini_pages)} Gemini tabs\n")

        if not gemini_pages:
            print("⚠️  No Gemini tabs found. Open some chats first!")
            return

        for i, page in enumerate(gemini_pages):
            print(f"{'='*60}")
            print(f"📖 Tab {i+1}/{len(gemini_pages)}")

            try:
                title = await page.title()
                print(f"   📝 {title[:60]}")

                # Scroll to ensure all content is loaded
                print("   📜 Scrolling to load content...")
                for _ in range(10):
                    try:
                        await page.evaluate('window.scrollBy(0, 1000)')
                        await asyncio.sleep(0.3)
                    except:
                        break

                # Scroll back to top
                try:
                    await page.evaluate('window.scrollTo(0, 0)')
                except:
                    pass

                # Extract content
                print("   📥 Extracting...")
                content = await extract_page_content(page)

                # Save JSON
                filename = f"{i+1:04d}_{sanitize_filename(title)}.json"
                filepath = OUTPUT_DIR / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)

                # Save raw text
                txt_filename = f"{i+1:04d}_{sanitize_filename(title)}.txt"
                txt_filepath = OUTPUT_DIR / txt_filename

                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"TITLE: {title}\n")
                    f.write(f"URL: {content['url']}\n")
                    f.write(f"EXTRACTED: {content['extracted_at']}\n")
                    f.write(f"{'='*60}\n\n")
                    f.write(content['raw_text'])

                # Save code blocks separately if substantial
                if content["code_blocks"]:
                    code_filename = f"{i+1:04d}_{sanitize_filename(title)}_code.txt"
                    code_filepath = OUTPUT_DIR / code_filename
                    with open(code_filepath, 'w', encoding='utf-8') as f:
                        for j, block in enumerate(content["code_blocks"]):
                            f.write(f"# === CODE BLOCK {j+1} ===\n")
                            f.write(block)
                            f.write("\n\n")

                manifest["pages"].append({
                    "number": i + 1,
                    "title": title,
                    "url": content["url"],
                    "filename": filename,
                    "chars": content["total_chars"],
                    "messages": content["message_count"],
                    "code_blocks": content["code_block_count"],
                    "math_blocks": content["math_block_count"],
                })

                manifest["total_pages"] += 1
                manifest["total_chars"] += content["total_chars"]
                manifest["total_code_blocks"] += content["code_block_count"]
                manifest["total_math_blocks"] += content["math_block_count"]

                print(f"   ✅ {content['total_chars']:,} chars, {content['code_block_count']} code blocks")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                manifest["pages"].append({
                    "number": i + 1,
                    "error": str(e),
                })

        # Save manifest
        manifest["extraction_completed"] = datetime.now().isoformat()
        with open(OUTPUT_DIR / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        # Create combined corpus file
        print(f"\n📚 Creating combined corpus...")
        corpus_path = OUTPUT_DIR / "full_corpus.txt"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for page_info in manifest["pages"]:
                if "error" not in page_info:
                    txt_file = OUTPUT_DIR / f"{page_info['number']:04d}_{sanitize_filename(page_info['title'])}.txt"
                    if txt_file.exists():
                        f.write(f"\n\n{'#'*80}\n")
                        f.write(f"# {page_info['title']}\n")
                        f.write(f"{'#'*80}\n\n")
                        f.write(txt_file.read_text(errors='ignore'))

        print(f"\n{'='*60}")
        print(f"🎉 EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"   Total pages: {manifest['total_pages']}")
        print(f"   Total chars: {manifest['total_chars']:,}")
        print(f"   Code blocks: {manifest['total_code_blocks']}")
        print(f"   Math blocks: {manifest['total_math_blocks']}")
        print(f"   Output: {OUTPUT_DIR}")
        print(f"   Combined: {corpus_path}")
        print(f"{'='*60}\n")

    return manifest


if __name__ == "__main__":
    asyncio.run(extract_all_tabs())
