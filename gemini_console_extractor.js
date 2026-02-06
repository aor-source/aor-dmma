/**
 * Gemini Research Extractor - Console Version
 * =============================================
 * Paste this directly into Chrome DevTools console on gemini.google.com
 * No Playwright, no external tools - runs natively in browser.
 *
 * USAGE:
 * 1. Go to gemini.google.com
 * 2. Open DevTools (Cmd+Option+I)
 * 3. Go to Console tab
 * 4. Paste this entire script
 * 5. Press Enter
 * 6. Wait for extraction
 * 7. Files auto-download
 */

(async function extractGeminiResearch() {
    console.log('🔬 Gemini Research Extractor starting...');

    // Research keywords for scoring
    const KEYWORDS = {
        'dead neuron': 10, 'dying relu': 10, 'gradient': 5, 'backprop': 5,
        'aesthetic': 10, 'qualia': 10, 'phenomenal': 7, 'beauty': 5,
        'emotion': 8, 'affect': 6, 'valence': 8, 'arousal': 8,
        'emoji': 10, 'semantic': 6, 'embedding': 5,
        'deepfake': 10, 'synthetic': 6, 'GAN': 7, 'forgery': 8,
        'equation': 6, 'theorem': 8, 'proof': 7, 'axiom': 8, 'entropy': 7,
        'architect': 8, 'AoR': 8, 'rhyme': 5,
        'script': 4, 'algorithm': 6, 'code': 3, 'python': 4,
        'transformer': 7, 'attention': 6, 'neural': 5, 'network': 4,
    };

    function scoreText(text) {
        const lower = text.toLowerCase();
        let score = 0;
        const matches = [];
        for (const [kw, weight] of Object.entries(KEYWORDS)) {
            const regex = new RegExp(kw, 'gi');
            const count = (lower.match(regex) || []).length;
            if (count > 0) {
                score += count * weight;
                matches.push(`${kw}(${count})`);
            }
        }
        return { score, matches };
    }

    // Find all chat links in sidebar
    function findChatLinks() {
        const selectors = [
            'a[href*="/app/"]',
            '[class*="conversation"] a',
            'aside a[href*="gemini"]',
            'nav a[href*="app"]',
        ];

        for (const sel of selectors) {
            const links = [...document.querySelectorAll(sel)];
            const valid = links.filter(a => a.href && a.href.includes('/app/'));
            if (valid.length > 2) {
                console.log(`Found ${valid.length} chats with selector: ${sel}`);
                return valid;
            }
        }
        return [];
    }

    // Scroll sidebar to load all chats
    async function loadAllChats() {
        console.log('📜 Scrolling sidebar to load all chats...');

        const sidebar = document.querySelector('aside, nav, [class*="sidebar"]');
        let prevCount = 0;
        let stable = 0;

        for (let i = 0; i < 50; i++) {
            const links = findChatLinks();
            const count = links.length;

            if (count === prevCount) {
                stable++;
                if (stable >= 3) break;
            } else {
                stable = 0;
                if (count % 20 === 0) console.log(`   Loaded ${count} chats...`);
            }
            prevCount = count;

            if (sidebar) sidebar.scrollTop += 500;
            window.scrollBy(0, 300);
            await new Promise(r => setTimeout(r, 500));
        }

        console.log(`   ✅ Found ${prevCount} total chats`);
        return prevCount;
    }

    // Extract content from current page
    function extractCurrentPage() {
        const content = {
            url: window.location.href,
            title: document.title,
            messages: [],
            codeBlocks: [],
            rawText: '',
        };

        // Get all text from main content
        const main = document.querySelector('main') || document.body;
        content.rawText = main.innerText;

        // Get structured messages
        const msgSelectors = ['[class*="message"]', '[class*="response"]', '[class*="turn"]', 'article'];
        for (const sel of msgSelectors) {
            const msgs = [...document.querySelectorAll(sel)];
            if (msgs.length >= 2) {
                content.messages = msgs.map(m => m.innerText).filter(t => t.length > 20);
                break;
            }
        }

        // Get code blocks
        const codeEls = document.querySelectorAll('pre, code, [class*="code"]');
        content.codeBlocks = [...codeEls].map(c => c.innerText).filter(t => t.length > 30);

        return content;
    }

    // Download helper
    function downloadFile(content, filename) {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Main extraction
    try {
        const results = {
            extracted: new Date().toISOString(),
            chats: [],
            highValue: [],
            totalChars: 0,
        };

        // Load all chats first
        await loadAllChats();
        const chatLinks = findChatLinks();

        console.log(`\n🔍 Scanning ${chatLinks.length} chats for research content...\n`);

        for (let i = 0; i < chatLinks.length; i++) {
            const link = chatLinks[i];
            const chatNum = i + 1;

            if (chatNum % 10 === 0) {
                console.log(`   Scanning ${chatNum}/${chatLinks.length}...`);
            }

            try {
                // Click to open chat
                link.click();
                await new Promise(r => setTimeout(r, 1500));

                // Scroll to load content
                for (let j = 0; j < 5; j++) {
                    window.scrollBy(0, 800);
                    await new Promise(r => setTimeout(r, 200));
                }
                window.scrollTo(0, 0);

                // Extract
                const content = extractCurrentPage();
                const { score, matches } = scoreText(content.rawText);

                const chatData = {
                    number: chatNum,
                    title: content.title,
                    url: content.url,
                    score: score,
                    matches: matches.slice(0, 10),
                    charCount: content.rawText.length,
                    codeBlocks: content.codeBlocks.length,
                    rawText: content.rawText,
                };

                results.chats.push(chatData);
                results.totalChars += content.rawText.length;

                if (score >= 30) {
                    results.highValue.push(chatData);
                    console.log(`   🎯 HIGH VALUE [${score}]: ${content.title.slice(0, 50)}`);
                }

            } catch (e) {
                console.warn(`   ⚠️ Error on chat ${chatNum}:`, e.message);
                results.chats.push({ number: chatNum, error: e.message });
            }

            // Small delay
            await new Promise(r => setTimeout(r, 300));
        }

        // Sort by score
        results.chats.sort((a, b) => (b.score || 0) - (a.score || 0));
        results.highValue.sort((a, b) => (b.score || 0) - (a.score || 0));

        console.log(`\n${'='.repeat(50)}`);
        console.log('🎉 EXTRACTION COMPLETE');
        console.log(`${'='.repeat(50)}`);
        console.log(`   Total chats: ${chatLinks.length}`);
        console.log(`   High value: ${results.highValue.length}`);
        console.log(`   Total chars: ${results.totalChars.toLocaleString()}`);

        // Download results
        const timestamp = new Date().toISOString().split('T')[0];

        // Download summary
        let summary = `GEMINI RESEARCH EXTRACTION\n${'='.repeat(60)}\n`;
        summary += `Extracted: ${results.extracted}\n`;
        summary += `Total chats: ${results.chats.length}\n`;
        summary += `High value: ${results.highValue.length}\n`;
        summary += `Total chars: ${results.totalChars.toLocaleString()}\n\n`;
        summary += `${'='.repeat(60)}\nHIGH VALUE RESEARCH CHATS:\n${'='.repeat(60)}\n\n`;

        for (const chat of results.highValue) {
            summary += `📌 [${chat.score}] ${chat.title}\n`;
            summary += `   Matches: ${chat.matches.join(', ')}\n`;
            summary += `   Chars: ${chat.charCount.toLocaleString()}\n\n`;
        }

        downloadFile(summary, `gemini_research_summary_${timestamp}.txt`);
        console.log('📥 Downloaded: summary');

        // Download full corpus
        let corpus = '';
        for (const chat of results.chats) {
            if (chat.rawText) {
                corpus += `\n\n${'#'.repeat(80)}\n`;
                corpus += `# ${chat.title}\n`;
                corpus += `# Score: ${chat.score || 0}\n`;
                corpus += `${'#'.repeat(80)}\n\n`;
                corpus += chat.rawText;
            }
        }
        downloadFile(corpus, `gemini_research_corpus_${timestamp}.txt`);
        console.log('📥 Downloaded: full corpus');

        // Download JSON
        downloadFile(JSON.stringify(results, null, 2), `gemini_research_${timestamp}.json`);
        console.log('📥 Downloaded: JSON data');

        console.log(`\n✅ All files downloaded to your Downloads folder`);

        return results;

    } catch (error) {
        console.error('❌ Extraction failed:', error);
    }
})();
