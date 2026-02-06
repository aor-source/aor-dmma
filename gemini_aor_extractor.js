/**
 * AoR Conversation Extractor for Gemini
 * ======================================
 *
 * Run this in Chrome DevTools console while logged into gemini.google.com
 *
 * INSTRUCTIONS:
 * 1. Go to https://gemini.google.com
 * 2. Open DevTools (Cmd+Option+I)
 * 3. Go to Console tab
 * 4. Paste this entire script
 * 5. Press Enter
 * 6. Wait for extraction to complete
 * 7. File will auto-download as aor_gemini_conversations.jsonl
 */

(async function extractGeminiAoR() {
    console.log('🎤 AoR Gemini Extractor starting...');

    // Patterns to identify AoR content
    const isAoRContent = (text) => {
        if (!text) return false;
        const patterns = [
            /\[Verse\s*\d*\]/i,
            /\[Hook\]/i,
            /\[Chorus\]/i,
            /\[Bridge\]/i,
            /Architect of Rhyme/i,
            /I am the Architect/i,
            /— AoR/i,
            /\[Intro\]/i,
            /\[Outro\]/i,
        ];
        return patterns.some(p => p.test(text));
    };

    // Parse song structure
    const parseStructure = (text) => {
        const verses = [];
        const hooks = [];
        const bridges = [];

        const sections = text.split(/(\[(?:Verse|Hook|Chorus|Bridge|Intro|Outro)\s*\d*\])/gi);
        let currentSection = 'intro';
        let currentContent = [];

        for (const part of sections) {
            if (/\[(Verse|Hook|Chorus|Bridge|Intro|Outro)\s*\d*\]/i.test(part)) {
                if (currentContent.length > 0) {
                    const content = currentContent.join('\n').trim();
                    if (currentSection.toLowerCase().includes('verse')) {
                        verses.push(content);
                    } else if (currentSection.toLowerCase().includes('hook') ||
                               currentSection.toLowerCase().includes('chorus')) {
                        hooks.push(content);
                    } else if (currentSection.toLowerCase().includes('bridge')) {
                        bridges.push(content);
                    }
                }
                currentSection = part;
                currentContent = [];
            } else {
                currentContent.push(part);
            }
        }

        // Save final section
        if (currentContent.length > 0) {
            const content = currentContent.join('\n').trim();
            if (currentSection.toLowerCase().includes('verse')) {
                verses.push(content);
            } else if (currentSection.toLowerCase().includes('hook') ||
                       currentSection.toLowerCase().includes('chorus')) {
                hooks.push(content);
            } else if (currentSection.toLowerCase().includes('bridge')) {
                bridges.push(content);
            }
        }

        return { verses, hooks, bridges };
    };

    // Count bars
    const countBars = (text) => {
        const lines = text.split('\n').filter(l => {
            const t = l.trim();
            return t.length > 10 && t.length < 150 && !t.startsWith('[');
        });
        return lines.length;
    };

    // Extract title
    const extractTitle = (text) => {
        const titleMatch = text.match(/\[Title:\s*([^\]]+)\]/) ||
                           text.match(/\*\*([^*]+)\*\*/) ||
                           text.match(/"([^"]+)"/);
        if (titleMatch) return titleMatch[1];

        const lines = text.split('\n').filter(l => l.trim() && !l.startsWith('['));
        if (lines[0]) return `Untitled (${lines[0].substring(0, 40)}...)`;
        return 'Untitled';
    };

    // METHOD 1: Try to scrape from DOM (visible conversations)
    const scrapeFromDOM = () => {
        const entries = [];
        console.log('📜 Attempting DOM extraction...');

        // Gemini uses various classes - try common patterns
        const messageSelectors = [
            '[data-message-id]',
            '.model-response-text',
            '.response-content',
            '.message-content',
            'message-content',
            '[class*="response"]',
            '[class*="message"]',
        ];

        let messages = [];
        for (const selector of messageSelectors) {
            const found = document.querySelectorAll(selector);
            if (found.length > 0) {
                messages = Array.from(found);
                console.log(`Found ${messages.length} elements with selector: ${selector}`);
                break;
            }
        }

        // Also try finding all text blocks
        if (messages.length === 0) {
            const allDivs = document.querySelectorAll('div');
            messages = Array.from(allDivs).filter(div => {
                const text = div.textContent || '';
                return text.length > 200 && isAoRContent(text);
            });
            console.log(`Found ${messages.length} divs with AoR content`);
        }

        for (const msg of messages) {
            const text = msg.textContent || msg.innerText || '';
            if (!isAoRContent(text)) continue;

            const structure = parseStructure(text);
            entries.push({
                id: `gemini_dom_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                title: extractTitle(text),
                timestamp: new Date().toISOString(),
                structure: {
                    verse_count: structure.verses.length,
                    has_hook: structure.hooks.length > 0,
                    has_bridge: structure.bridges.length > 0,
                    bar_count: countBars(text),
                },
                verses: structure.verses,
                hooks: structure.hooks,
                bridges: structure.bridges,
                raw_text: text.substring(0, 10000),
                source: 'gemini.google.com'
            });
        }

        return entries;
    };

    // METHOD 2: Check for Gemini's internal API
    const tryAPIExtraction = async () => {
        console.log('🔌 Attempting API extraction...');

        // Gemini uses different endpoint patterns
        const endpoints = [
            '/_/BardChatUi/data/batchexecute',
            '/api/conversations',
            '/api/history',
        ];

        // Try to intercept or find API data in page state
        const scripts = document.querySelectorAll('script');
        let conversationData = null;

        for (const script of scripts) {
            const content = script.textContent || '';
            if (content.includes('conversation') || content.includes('response')) {
                // Look for JSON-like structures
                const jsonMatches = content.match(/\{[^{}]*"(?:conversation|responses|messages)"[^{}]*\}/g);
                if (jsonMatches) {
                    for (const match of jsonMatches) {
                        try {
                            const parsed = JSON.parse(match);
                            if (parsed.conversations || parsed.responses || parsed.messages) {
                                conversationData = parsed;
                                break;
                            }
                        } catch (e) {}
                    }
                }
            }
        }

        if (conversationData) {
            console.log('Found embedded conversation data');
            return conversationData;
        }

        return null;
    };

    // METHOD 3: Google Takeout instructions
    const showTakeoutInstructions = () => {
        console.log(`
╔══════════════════════════════════════════════════════════════════╗
║              GOOGLE TAKEOUT EXTRACTION METHOD                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  For complete Gemini history, use Google Takeout:                 ║
║                                                                   ║
║  1. Go to: https://takeout.google.com                            ║
║  2. Click "Deselect all"                                         ║
║  3. Scroll down and select "Gemini Apps"                         ║
║  4. Click "Next step"                                            ║
║  5. Choose export frequency: "Export once"                       ║
║  6. Choose file type: ".zip"                                     ║
║  7. Click "Create export"                                        ║
║  8. Wait for email with download link                            ║
║  9. Download and unzip                                           ║
║  10. Find conversations in: Takeout/Gemini Apps/                 ║
║                                                                   ║
║  Then run:                                                        ║
║    python3 ~/aor-dmma/parse_gemini_takeout.py <path-to-folder>   ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
`);
    };

    // Main extraction
    try {
        let allEntries = [];

        // Try DOM scraping first
        const domEntries = scrapeFromDOM();
        if (domEntries.length > 0) {
            console.log(`✅ DOM extraction found ${domEntries.length} AoR tracks`);
            allEntries.push(...domEntries);
        }

        // Try API extraction
        const apiData = await tryAPIExtraction();
        if (apiData) {
            console.log('✅ Found API data (parsing...)');
            // Would need to parse based on actual structure
        }

        if (allEntries.length > 0) {
            // Create JSONL and download
            const jsonl = allEntries.map(e => JSON.stringify(e)).join('\n');
            const blob = new Blob([jsonl], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `aor_gemini_${new Date().toISOString().split('T')[0]}.jsonl`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            console.log(`\n📥 Downloaded: ${a.download}`);
            console.log(`   Tracks extracted: ${allEntries.length}`);
        } else {
            console.log('\n⚠️ No AoR content found in visible page.');
            console.log('   This might mean:');
            console.log('   1. You need to scroll through conversations to load them');
            console.log('   2. The conversations are in a different format');
            console.log('   3. Use Google Takeout for complete history');
        }

        // Always show Takeout option for complete extraction
        showTakeoutInstructions();

        console.log('\nMove downloaded files to ~/aor-dmma/aor_catalog/ and run:');
        console.log('  python3 merge_catalogs.py');

        return allEntries;

    } catch (error) {
        console.error('❌ Extraction error:', error);
        showTakeoutInstructions();
    }
})();
