/**
 * AoR Conversation Extractor for Claude.ai
 * =========================================
 *
 * Run this in Chrome DevTools console while logged into claude.ai
 *
 * INSTRUCTIONS:
 * 1. Go to https://claude.ai
 * 2. Open DevTools (Cmd+Option+I)
 * 3. Go to Console tab
 * 4. Paste this entire script
 * 5. Press Enter
 * 6. Wait for extraction to complete
 * 7. File will auto-download as aor_conversations.jsonl
 */

(async function extractAoRConversations() {
    console.log('🎤 AoR Conversation Extractor starting...');

    // Get auth token from local storage or cookies
    const getAuthToken = () => {
        // Try to get from cookies
        const cookies = document.cookie.split(';').reduce((acc, c) => {
            const [key, val] = c.trim().split('=');
            acc[key] = val;
            return acc;
        }, {});
        return cookies['sessionKey'] || null;
    };

    // Get organization ID from URL or page
    const getOrgId = () => {
        const match = window.location.pathname.match(/\/organization\/([^\/]+)/);
        if (match) return match[1];
        // Try to extract from page
        const orgElement = document.querySelector('[data-org-id]');
        return orgElement?.dataset?.orgId || null;
    };

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
        ];
        return patterns.some(p => p.test(text));
    };

    // Extract conversation list
    const fetchConversationList = async () => {
        console.log('📋 Fetching conversation list...');

        // Use the internal API
        const response = await fetch('/api/organizations/'+getOrgId()+'/chat_conversations', {
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            // Try alternate endpoint
            const altResponse = await fetch('/api/chat_conversations', {
                credentials: 'include',
            });
            if (altResponse.ok) {
                return await altResponse.json();
            }
            throw new Error(`Failed to fetch conversations: ${response.status}`);
        }

        return await response.json();
    };

    // Fetch full conversation
    const fetchConversation = async (convId) => {
        const response = await fetch(`/api/organizations/${getOrgId()}/chat_conversations/${convId}`, {
            credentials: 'include',
        });

        if (!response.ok) {
            // Try alternate
            const alt = await fetch(`/api/chat_conversations/${convId}`, {
                credentials: 'include',
            });
            if (alt.ok) return await alt.json();
            return null;
        }

        return await response.json();
    };

    // Parse conversation for AoR content
    const parseConversation = (conv) => {
        const entries = [];

        if (!conv || !conv.chat_messages) return entries;

        for (const msg of conv.chat_messages) {
            if (msg.sender !== 'assistant') continue;

            const text = msg.text || '';
            if (!isAoRContent(text)) continue;

            // Extract structure
            const verses = [];
            const hooks = [];
            const bridges = [];

            // Split by section markers
            const sections = text.split(/(\[(?:Verse|Hook|Chorus|Bridge|Intro|Outro)\s*\d*\])/gi);
            let currentSection = 'intro';
            let currentContent = [];

            for (const part of sections) {
                if (/\[(Verse|Hook|Chorus|Bridge|Intro|Outro)\s*\d*\]/i.test(part)) {
                    // Save previous
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

            // Count bars (lyric lines)
            const lines = text.split('\n').filter(l => {
                const t = l.trim();
                return t.length > 10 && t.length < 150 && !t.startsWith('[');
            });

            // Extract title
            let title = 'Untitled';
            const titleMatch = text.match(/\[Title:\s*([^\]]+)\]/) ||
                               text.match(/\*\*([^*]+)\*\*/) ||
                               text.match(/"([^"]+)"/);
            if (titleMatch) title = titleMatch[1];
            else if (lines[0]) title = `Untitled (${lines[0].substring(0, 40)}...)`;

            entries.push({
                id: msg.uuid || `${conv.uuid}_${Date.now()}`,
                conversation_id: conv.uuid,
                conversation_name: conv.name || 'Untitled Conversation',
                title: title,
                timestamp: msg.created_at || conv.created_at,
                structure: {
                    verse_count: verses.length,
                    has_hook: hooks.length > 0,
                    has_bridge: bridges.length > 0,
                    bar_count: lines.length,
                },
                verses: verses,
                hooks: hooks,
                bridges: bridges,
                raw_text: text.substring(0, 10000),
                source: 'claude.ai'
            });
        }

        return entries;
    };

    // Main extraction
    try {
        const allEntries = [];
        let conversationsProcessed = 0;
        let aorTracksFound = 0;

        // Get conversation list
        let conversations;
        try {
            conversations = await fetchConversationList();
        } catch (e) {
            console.error('❌ Could not fetch conversation list. Trying alternate method...');

            // Manual method - scrape from sidebar
            const convLinks = document.querySelectorAll('a[href*="/chat/"]');
            conversations = Array.from(convLinks).map(a => ({
                uuid: a.href.split('/chat/')[1]?.split('?')[0],
                name: a.textContent
            })).filter(c => c.uuid);

            if (conversations.length === 0) {
                throw new Error('No conversations found. Make sure you are on claude.ai');
            }
        }

        console.log(`📚 Found ${conversations.length} conversations`);

        // Process each conversation
        for (const conv of conversations) {
            try {
                conversationsProcessed++;

                if (conversationsProcessed % 10 === 0) {
                    console.log(`⏳ Processing ${conversationsProcessed}/${conversations.length}...`);
                }

                const fullConv = await fetchConversation(conv.uuid);
                if (!fullConv) continue;

                const entries = parseConversation(fullConv);
                if (entries.length > 0) {
                    allEntries.push(...entries);
                    aorTracksFound += entries.length;
                }

                // Rate limiting
                await new Promise(r => setTimeout(r, 100));

            } catch (e) {
                console.warn(`⚠️ Error processing conversation ${conv.uuid}:`, e);
            }
        }

        console.log(`\n✅ Extraction complete!`);
        console.log(`   Conversations processed: ${conversationsProcessed}`);
        console.log(`   AoR tracks found: ${aorTracksFound}`);

        // Create JSONL and download
        const jsonl = allEntries.map(e => JSON.stringify(e)).join('\n');
        const blob = new Blob([jsonl], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `aor_conversations_${new Date().toISOString().split('T')[0]}.jsonl`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log(`\n📥 Downloaded: ${a.download}`);
        console.log('\nMove this file to ~/aor-dmma/aor_catalog/ and run:');
        console.log('  python3 aor_context_index.py build');

        return allEntries;

    } catch (error) {
        console.error('❌ Extraction failed:', error);
        console.log('\nTROUBLESHOOTING:');
        console.log('1. Make sure you are logged into claude.ai');
        console.log('2. Refresh the page and try again');
        console.log('3. Check the Network tab for any blocked requests');
    }
})();
