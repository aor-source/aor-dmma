#!/usr/bin/env python3
"""
AAVE Knowledge Expansion Tool
=============================
Expands the AAVE lexicon by scraping web resources and integrating new terms.

Features:
- Scrapes AAVE dictionaries and glossaries from the web
- Validates terms against existing lexicon
- Adds new terms with proper categorization
- Re-indexes the corpus with expanded knowledge

Usage:
    python3 expand_knowledge.py scrape              # Scrape web resources
    python3 expand_knowledge.py add "term" "cat"    # Add term to category
    python3 expand_knowledge.py validate            # Validate lexicon against corpus
    python3 expand_knowledge.py reindex             # Rebuild index with new terms
    python3 expand_knowledge.py stats               # Show lexicon statistics
    python3 expand_knowledge.py search "query"      # Web search for AAVE resources

Authors: Jon + Claude - February 2026
"""

import os
# Set TF-IDF mode before imports
os.environ["USE_TFIDF"] = "1"
os.environ["DISABLE_FAISS"] = "1"

import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
from urllib.parse import urljoin, urlparse
import hashlib

# Web scraping imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Local imports
SCRIPT_DIR = Path(__file__).parent
LEXICON_PATH = SCRIPT_DIR / "aave_lexicon.json"
LEXICON_BACKUP = SCRIPT_DIR / "aave_lexicon_backup.json"
EXPANSION_LOG = SCRIPT_DIR / "expansion_log.json"

# Known AAVE resource URLs (curated list)
AAVE_RESOURCES = [
    {
        "name": "Wikipedia - List of English words from African American sources",
        "url": "https://en.wikipedia.org/wiki/List_of_English_words_of_African_origin",
        "type": "encyclopedia"
    },
    {
        "name": "Urban Dictionary - AAVE Tag",
        "url": "https://www.urbandictionary.com/define.php?term=AAVE",
        "type": "dictionary"
    },
]

# Additional skip words for filtering (common words that aren't AAVE-specific)
SKIP_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
    'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
    'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
    'did', 'oil', 'sit', 'set', 'african', 'american', 'english', 'language',
    'vernacular', 'dialect', 'linguistics', 'grammar', 'speech', 'black',
    'white', 'racial', 'ethnic', 'culture', 'history', 'movement', 'people',
    'united', 'states', 'century', 'years', 'time', 'world', 'also', 'more',
    'other', 'some', 'such', 'than', 'them', 'then', 'these', 'they', 'this',
    'from', 'have', 'been', 'been', 'would', 'could', 'should', 'will', 'just',
    'make', 'like', 'know', 'take', 'come', 'want', 'look', 'give', 'use',
    'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call',
    'self', 'back', 'pan', 'anglo', 'afro', 'down', 'national', 'fort',
    'council', 'romani', 'dallas', 'americo', 'hellenic', 'liberians', 'low'
}

# Search engines and APIs for finding AAVE resources
SEARCH_QUERIES = [
    "AAVE vocabulary list",
    "African American Vernacular English glossary",
    "hip hop slang dictionary",
    "Black English terms",
    "AAVE linguistic features",
]


class AAVEKnowledgeExpander:
    """Expands AAVE lexicon with web-sourced terms and validation."""

    def __init__(self):
        self.lexicon = self._load_lexicon()
        self.expansion_log = self._load_expansion_log()
        self.new_terms = []
        self.session = None
        if HAS_REQUESTS:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AAVE-Research-Bot/1.0"
            })

    def _load_lexicon(self) -> Dict:
        """Load the AAVE lexicon."""
        if LEXICON_PATH.exists():
            with open(LEXICON_PATH) as f:
                return json.load(f)
        return {}

    def _load_expansion_log(self) -> Dict:
        """Load the expansion log tracking changes."""
        if EXPANSION_LOG.exists():
            with open(EXPANSION_LOG) as f:
                return json.load(f)
        return {"expansions": [], "sources": [], "last_update": None}

    def _save_lexicon(self):
        """Save lexicon with backup."""
        # Create backup
        if LEXICON_PATH.exists():
            import shutil
            shutil.copy(LEXICON_PATH, LEXICON_BACKUP)

        # Save updated lexicon
        self.lexicon["metadata"]["last_expanded"] = datetime.now().isoformat()
        self.lexicon["metadata"]["version"] = self.lexicon["metadata"].get("version", "3.0") + "-expanded"

        with open(LEXICON_PATH, 'w') as f:
            json.dump(self.lexicon, f, indent=2)

        print(f"[+] Lexicon saved: {LEXICON_PATH}")
        print(f"[+] Backup saved: {LEXICON_BACKUP}")

    def _save_expansion_log(self):
        """Save the expansion log."""
        self.expansion_log["last_update"] = datetime.now().isoformat()
        with open(EXPANSION_LOG, 'w') as f:
            json.dump(self.expansion_log, f, indent=2)

    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content with rate limiting and error handling."""
        if not HAS_REQUESTS:
            print("[!] requests library not installed: pip install requests")
            return None

        try:
            time.sleep(1)  # Rate limiting
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"[!] Failed to fetch {url}: {e}")
            return None

    def _extract_terms_from_html(self, html: str, url: str) -> List[Dict]:
        """Extract potential AAVE terms from HTML content."""
        if not HAS_BS4:
            print("[!] BeautifulSoup not installed: pip install beautifulsoup4")
            return []

        soup = BeautifulSoup(html, 'html.parser')
        terms = []

        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # Look for definition lists (common in dictionaries)
        for dl in soup.find_all('dl'):
            for dt in dl.find_all('dt'):
                term = dt.get_text().strip().lower()
                dd = dt.find_next_sibling('dd')
                definition = dd.get_text().strip() if dd else ""
                if self._is_valid_term(term):
                    terms.append({
                        "term": term,
                        "definition": definition[:200],
                        "source": url
                    })

        # Look for list items with potential terms
        for li in soup.find_all('li'):
            text = li.get_text().strip()
            # Pattern: "term - definition" or "term: definition"
            match = re.match(r'^([a-z\']+(?:\s+[a-z\']+)?)\s*[-:]\s*(.+)$', text.lower())
            if match:
                term, definition = match.groups()
                if self._is_valid_term(term):
                    terms.append({
                        "term": term.strip(),
                        "definition": definition[:200],
                        "source": url
                    })

        # Look for bold/strong terms followed by text
        for bold in soup.find_all(['b', 'strong']):
            term = bold.get_text().strip().lower()
            if self._is_valid_term(term) and len(term) < 30:
                next_text = bold.next_sibling
                definition = next_text.strip() if next_text and isinstance(next_text, str) else ""
                terms.append({
                    "term": term,
                    "definition": definition[:200],
                    "source": url
                })

        return terms

    def _is_valid_term(self, term: str) -> bool:
        """Check if a term is a valid AAVE candidate."""
        if not term or len(term) < 2 or len(term) > 40:
            return False
        # Must contain mostly letters
        if not re.match(r"^[a-z' -]+$", term.lower()):
            return False
        # Skip common English words that aren't likely AAVE-specific
        if term.lower() in SKIP_WORDS:
            return False
        # Skip single common words
        if len(term) < 4 and term.lower() not in {'yo', 'og', 'gg', 'nah', 'cap', 'lit', 'drip'}:
            return False
        return True

    def _is_already_in_lexicon(self, term: str) -> bool:
        """Check if term already exists in lexicon."""
        term_lower = term.lower()

        # Check semantic inversions
        if term_lower in self.lexicon.get("semantic_inversions", {}):
            return True

        # Check contractions
        if term_lower in self.lexicon.get("contractions", []):
            return True

        # Check all slang categories
        for category, terms in self.lexicon.get("slang_terms", {}).items():
            if term_lower in terms:
                return True

        # Check regional variants
        for region, terms in self.lexicon.get("regional_variants", {}).items():
            if term_lower in terms:
                return True

        return False

    def _categorize_term(self, term: str, definition: str = "") -> str:
        """Attempt to categorize a term based on definition or patterns."""
        definition_lower = definition.lower()
        term_lower = term.lower()

        # Check for category indicators in definition
        if any(w in definition_lower for w in ['good', 'great', 'excellent', 'positive', 'approval']):
            return "positive"
        if any(w in definition_lower for w in ['bad', 'negative', 'dislike', 'insult', 'derogatory']):
            return "negative"
        if any(w in definition_lower for w in ['money', 'cash', 'dollar', 'rich', 'wealth']):
            return "money_terms"
        if any(w in definition_lower for w in ['drug', 'weed', 'marijuana', 'high', 'smoke']):
            return "drugs_alcohol"
        if any(w in definition_lower for w in ['car', 'drive', 'wheel', 'ride', 'vehicle']):
            return "vehicles"
        if any(w in definition_lower for w in ['friend', 'person', 'man', 'woman', 'homie']):
            return "people"
        if any(w in definition_lower for w in ['music', 'rap', 'hip hop', 'beat', 'song']):
            return "music_performance"
        if any(w in definition_lower for w in ['very', 'really', 'extremely', 'so much']):
            return "intensifiers"
        if any(w in definition_lower for w in ['place', 'location', 'area', 'neighborhood']):
            return "locations"

        # Default to general
        return "general"

    def scrape_resources(self) -> List[Dict]:
        """Scrape known AAVE resources for new terms."""
        print("\n[*] Scraping AAVE Resources...")
        all_terms = []

        for resource in AAVE_RESOURCES:
            print(f"\n  Fetching: {resource['name']}")
            html = self._fetch_url(resource['url'])
            if html:
                terms = self._extract_terms_from_html(html, resource['url'])
                print(f"    Found {len(terms)} potential terms")

                # Filter out existing terms
                new_terms = [t for t in terms if not self._is_already_in_lexicon(t['term'])]
                print(f"    {len(new_terms)} are new")
                all_terms.extend(new_terms)

                # Log source
                self.expansion_log["sources"].append({
                    "url": resource['url'],
                    "name": resource['name'],
                    "scraped_at": datetime.now().isoformat(),
                    "terms_found": len(terms),
                    "new_terms": len(new_terms)
                })

        # Deduplicate
        seen = set()
        unique_terms = []
        for t in all_terms:
            if t['term'] not in seen:
                seen.add(t['term'])
                unique_terms.append(t)

        print(f"\n[+] Total unique new terms found: {len(unique_terms)}")
        return unique_terms

    def add_term(self, term: str, category: str, definition: str = "") -> bool:
        """Add a single term to the lexicon."""
        term_lower = term.lower().strip()

        if self._is_already_in_lexicon(term_lower):
            print(f"[!] Term '{term}' already exists in lexicon")
            return False

        # Validate category
        valid_categories = list(self.lexicon.get("slang_terms", {}).keys())
        if category not in valid_categories:
            print(f"[!] Invalid category: {category}")
            print(f"    Valid categories: {', '.join(valid_categories)}")
            return False

        # Add to lexicon
        if "slang_terms" not in self.lexicon:
            self.lexicon["slang_terms"] = {}
        if category not in self.lexicon["slang_terms"]:
            self.lexicon["slang_terms"][category] = []

        self.lexicon["slang_terms"][category].append(term_lower)

        # Log addition
        self.expansion_log["expansions"].append({
            "term": term_lower,
            "category": category,
            "definition": definition,
            "added_at": datetime.now().isoformat(),
            "source": "manual"
        })

        print(f"[+] Added '{term}' to category '{category}'")
        return True

    def add_terms_batch(self, terms: List[Dict]) -> int:
        """Add multiple terms in batch."""
        added = 0
        for t in terms:
            term = t['term']
            definition = t.get('definition', '')
            category = self._categorize_term(term, definition)

            if self.add_term(term, category, definition):
                added += 1

        if added > 0:
            self._save_lexicon()
            self._save_expansion_log()

        return added

    def validate_against_corpus(self) -> Dict:
        """Validate lexicon terms against the actual corpus."""
        from aor_context_index import AAVEContextIndex

        print("\n[*] Validating lexicon against corpus...")

        indexer = AAVEContextIndex()
        if not indexer.load_index():
            print("[!] No index found. Build first with: python3 aor_context_index.py build")
            return {}

        results = {
            "total_terms": 0,
            "found_in_corpus": 0,
            "not_found": [],
            "most_frequent": [],
            "validation_date": datetime.now().isoformat()
        }

        # Collect all terms
        all_terms = set()

        for term in self.lexicon.get("semantic_inversions", {}):
            all_terms.add(term)

        for terms in self.lexicon.get("slang_terms", {}).values():
            all_terms.update(terms)

        all_terms.update(self.lexicon.get("contractions", []))

        results["total_terms"] = len(all_terms)
        print(f"  Total lexicon terms: {len(all_terms)}")

        # Count term occurrences in corpus
        term_counts = {}
        corpus_text = " ".join(c["text"].lower() for c in indexer.metadata)

        for term in all_terms:
            count = corpus_text.count(term.lower())
            term_counts[term] = count
            if count > 0:
                results["found_in_corpus"] += 1
            else:
                results["not_found"].append(term)

        # Sort by frequency
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        results["most_frequent"] = [{"term": t, "count": c} for t, c in sorted_terms[:50]]

        print(f"  Terms found in corpus: {results['found_in_corpus']}")
        print(f"  Terms not found: {len(results['not_found'])}")

        return results

    def show_stats(self):
        """Display lexicon statistics."""
        print("\n" + "="*60)
        print("  AAVE LEXICON STATISTICS")
        print("="*60)

        # Count terms by category
        print("\n[Semantic Inversions]")
        inversions = self.lexicon.get("semantic_inversions", {})
        print(f"  Count: {len(inversions)}")
        print(f"  Examples: {', '.join(list(inversions.keys())[:10])}")

        print("\n[Contractions]")
        contractions = self.lexicon.get("contractions", [])
        print(f"  Count: {len(contractions)}")
        print(f"  Examples: {', '.join(contractions[:10])}")

        print("\n[Slang Terms by Category]")
        slang = self.lexicon.get("slang_terms", {})
        total_slang = 0
        for category, terms in sorted(slang.items()):
            print(f"  {category}: {len(terms)}")
            total_slang += len(terms)
        print(f"  TOTAL: {total_slang}")

        print("\n[Regional Variants]")
        regional = self.lexicon.get("regional_variants", {})
        for region, terms in sorted(regional.items()):
            print(f"  {region}: {len(terms)}")

        print("\n[Classic Hip-Hop Terms]")
        classic = self.lexicon.get("classic_hip_hop_terms", {})
        for era, terms in sorted(classic.items()):
            print(f"  {era}: {len(terms)}")

        # Total
        total = (len(inversions) + len(contractions) + total_slang +
                sum(len(t) for t in regional.values()) +
                sum(len(t) for t in classic.values()))
        print(f"\n[GRAND TOTAL]: {total} terms")

        # Expansion log stats
        if self.expansion_log.get("expansions"):
            print(f"\n[Expansion History]")
            print(f"  Terms added: {len(self.expansion_log['expansions'])}")
            print(f"  Sources scraped: {len(self.expansion_log.get('sources', []))}")
            if self.expansion_log.get("last_update"):
                print(f"  Last update: {self.expansion_log['last_update']}")

    def web_search_resources(self, query: str) -> List[Dict]:
        """
        Search the web for AAVE resources.
        Note: This is a placeholder - real implementation would use a search API.
        """
        print(f"\n[*] Searching for: '{query}'")
        print("\n[!] Web search requires integration with a search API.")
        print("    Recommended options:")
        print("    - DuckDuckGo Instant Answer API (free)")
        print("    - Google Custom Search API (requires API key)")
        print("    - Bing Web Search API (requires API key)")
        print("\n    For now, use the curated resource list with 'scrape' command.")

        # Return curated resources as fallback
        print("\n[*] Curated AAVE resources:")
        for r in AAVE_RESOURCES:
            print(f"    - {r['name']}: {r['url']}")

        return AAVE_RESOURCES

    def reindex(self):
        """Rebuild the index with the updated lexicon."""
        print("\n[*] Rebuilding index with updated lexicon...")

        from aor_context_index import AAVEContextIndex
        indexer = AAVEContextIndex()
        success = indexer.build_index()

        if success:
            print("[+] Index rebuilt successfully")
        else:
            print("[!] Index rebuild failed")

        return success


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    expander = AAVEKnowledgeExpander()

    if cmd == "scrape":
        if not HAS_REQUESTS or not HAS_BS4:
            print("[!] Web scraping requires: pip install requests beautifulsoup4")
            return

        terms = expander.scrape_resources()
        if terms:
            print(f"\n[*] Found {len(terms)} new terms. Add them to lexicon? (y/n)")
            confirm = input().strip().lower()
            if confirm == 'y':
                added = expander.add_terms_batch(terms)
                print(f"[+] Added {added} terms to lexicon")
            else:
                print("[*] Terms not added. You can add manually with 'add' command.")
                print("\n[Sample terms found:]")
                for t in terms[:10]:
                    print(f"  - {t['term']}: {t.get('definition', '')[:60]}...")

    elif cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: expand_knowledge.py add 'term' 'category'")
            print("       Categories: positive, negative, intensifiers, money_terms, etc.")
            return
        term = sys.argv[2]
        category = sys.argv[3]
        definition = sys.argv[4] if len(sys.argv) > 4 else ""

        if expander.add_term(term, category, definition):
            expander._save_lexicon()
            expander._save_expansion_log()

    elif cmd == "validate":
        results = expander.validate_against_corpus()
        if results:
            print(f"\n[Validation Results]")
            print(f"  Coverage: {results['found_in_corpus']}/{results['total_terms']} "
                  f"({100*results['found_in_corpus']/results['total_terms']:.1f}%)")

            if results['not_found']:
                print(f"\n[Terms not found in corpus (sample)]")
                for t in results['not_found'][:20]:
                    print(f"  - {t}")

            if results['most_frequent']:
                print(f"\n[Most frequent terms in corpus]")
                for item in results['most_frequent'][:15]:
                    print(f"  {item['term']}: {item['count']}")

    elif cmd == "reindex":
        expander.reindex()

    elif cmd == "stats":
        expander.show_stats()

    elif cmd == "search":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "AAVE vocabulary"
        expander.web_search_resources(query)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
