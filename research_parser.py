#!/usr/bin/env python3
"""
Research Corpus Parser
======================
Extracts mathematical content, proofs, equations, and ideas from
Gemini research conversations.

Captures:
- LaTeX equations and proofs
- Mathematical formulas (inline and block)
- Algorithm descriptions
- Core theoretical ideas
- Code implementations
- Novel concepts

Usage:
    python3 research_parser.py <corpus_file>
    python3 research_parser.py ~/Downloads/gemini_research_corpus*.txt
    python3 research_parser.py --watch  # Watch Downloads for new files

Authors: Jon + Claude - February 2026
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib

OUTPUT_DIR = Path.home() / "aor-dmma" / "research_index"


# ============================================================================
# EXTRACTION PATTERNS
# ============================================================================

# LaTeX patterns
LATEX_PATTERNS = {
    # Display math
    'display_dollar': r'\$\$(.*?)\$\$',
    'display_bracket': r'\\\[(.*?)\\\]',
    'equation_env': r'\\begin\{equation\}(.*?)\\end\{equation\}',
    'align_env': r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
    'gather_env': r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}',

    # Inline math
    'inline_dollar': r'(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)',
    'inline_paren': r'\\\((.*?)\\\)',

    # Theorem environments
    'theorem': r'\\begin\{theorem\}(.*?)\\end\{theorem\}',
    'lemma': r'\\begin\{lemma\}(.*?)\\end\{lemma\}',
    'proof': r'\\begin\{proof\}(.*?)\\end\{proof\}',
    'definition': r'\\begin\{definition\}(.*?)\\end\{definition\}',
    'proposition': r'\\begin\{proposition\}(.*?)\\end\{proposition\}',
    'corollary': r'\\begin\{corollary\}(.*?)\\end\{corollary\}',
}

# Mathematical notation (non-LaTeX)
MATH_PATTERNS = {
    'fraction': r'\b(\d+)\s*/\s*(\d+)\b',
    'exponent': r'\b(\w+)\s*\^\s*(\d+|\{[^}]+\})',
    'subscript': r'\b(\w+)\s*_\s*(\d+|\{[^}]+\})',
    'summation': r'(?:sum|Σ|∑)\s*(?:from|_)?\s*(\w+)\s*(?:to|=)?\s*(\w+)',
    'integral': r'(?:integral|∫)\s*(?:from)?\s*(\w+)\s*(?:to)?\s*(\w+)',
    'derivative': r'(?:d|∂)(\w+)\s*/\s*(?:d|∂)(\w+)',
    'limit': r'lim\s*(?:_\{)?(\w+)\s*(?:->|→|\\to)\s*(\w+)',
    'infinity': r'(?:infinity|∞|\\infty)',
    'approx': r'(?:≈|\\approx|~=)',
    'proportional': r'(?:∝|\\propto)',
}

# Concept patterns
CONCEPT_PATTERNS = {
    'definition_inline': r'(?:define|defined as|let|denote)\s+(\w+)\s+(?:as|to be|=)',
    'theorem_inline': r'(?:theorem|lemma|proposition):\s*(.+?)(?:\.|$)',
    'if_then': r'if\s+(.+?)\s+then\s+(.+?)(?:\.|$)',
    'iff': r'(.+?)\s+(?:if and only if|iff|⟺)\s+(.+?)(?:\.|$)',
    'implies': r'(.+?)\s+(?:implies|⟹|=>)\s+(.+?)(?:\.|$)',
    'therefore': r'(?:therefore|thus|hence|∴)\s+(.+?)(?:\.|$)',
}

# Research topic identifiers
RESEARCH_TOPICS = {
    'dead_neuron': [
        r'dead\s*neuron', r'dying\s*relu', r'neuron\s*death',
        r'activation\s*collapse', r'gradient\s*vanish',
    ],
    'aesthetics': [
        r'aesthetic', r'beauty\s*function', r'qualia',
        r'phenomenal', r'subjective\s*experience',
    ],
    'emotion_targets': [
        r'emotion\s*target', r'affect\s*model', r'valence',
        r'arousal', r'sentiment\s*vector', r'feeling\s*space',
    ],
    'emoji_language': [
        r'emoji\s*(?:as|language|semantic)', r'emoticon\s*embedding',
        r'unicode\s*meaning', r'symbol\s*vector',
    ],
    'deepfake': [
        r'deepfake', r'fake\s*detection', r'synthetic\s*media',
        r'forgery', r'authenticity',
    ],
    'energy_efficiency': [
        r'energy\s*(?:efficiency|reduction)', r'compute\s*cost',
        r'node\s*energy', r'order\s*of\s*magnitude',
    ],
    'information_theory': [
        r'entropy', r'mutual\s*information', r'KL\s*divergence',
        r'information\s*bottleneck', r'channel\s*capacity',
    ],
}


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_latex(text):
    """Extract all LaTeX content"""
    results = {
        'display_math': [],
        'inline_math': [],
        'theorems': [],
        'proofs': [],
        'definitions': [],
    }

    # Display math
    for name, pattern in LATEX_PATTERNS.items():
        if 'display' in name or 'equation' in name or 'align' in name or 'gather' in name:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for m in matches:
                content = m.strip() if isinstance(m, str) else m[0].strip()
                if len(content) > 3:
                    results['display_math'].append({
                        'type': name,
                        'content': content,
                        'hash': hashlib.md5(content.encode()).hexdigest()[:8],
                    })

    # Inline math
    for name, pattern in LATEX_PATTERNS.items():
        if 'inline' in name:
            matches = re.findall(pattern, text, re.DOTALL)
            for m in matches:
                content = m.strip() if isinstance(m, str) else m[0].strip()
                if len(content) > 1 and len(content) < 200:
                    results['inline_math'].append(content)

    # Theorem environments
    for env in ['theorem', 'lemma', 'proposition', 'corollary']:
        pattern = LATEX_PATTERNS.get(env)
        if pattern:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for m in matches:
                results['theorems'].append({
                    'type': env,
                    'content': m.strip(),
                })

    # Proofs
    pattern = LATEX_PATTERNS.get('proof')
    if pattern:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for m in matches:
            results['proofs'].append(m.strip())

    # Definitions
    pattern = LATEX_PATTERNS.get('definition')
    if pattern:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for m in matches:
            results['definitions'].append(m.strip())

    return results


def extract_equations(text):
    """Extract mathematical equations and formulas"""
    equations = []

    # Look for equation-like patterns
    eq_patterns = [
        r'([A-Za-z_]\w*)\s*=\s*([^=\n]{10,100})',  # x = expression
        r'([A-Za-z_]\w*)\s*:=\s*([^=\n]{10,100})',  # x := expression (definition)
        r'∀\s*([^∃\n]{5,100})',  # Universal quantifier
        r'∃\s*([^∀\n]{5,100})',  # Existential quantifier
        r'([^=\n]{5,50})\s*≤\s*([^=\n]{5,50})',  # Inequalities
        r'([^=\n]{5,50})\s*≥\s*([^=\n]{5,50})',
    ]

    for pattern in eq_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if isinstance(m, tuple):
                eq = ' = '.join(m) if '=' not in str(m) else str(m)
            else:
                eq = m
            if len(eq) > 5:
                equations.append(eq.strip())

    return list(set(equations))


def extract_ideas(text):
    """Extract core ideas and concepts"""
    ideas = []

    # Key phrases that often introduce ideas
    idea_markers = [
        r'(?:key insight|main idea|core concept|fundamental):\s*([^.]+\.)',
        r'(?:we propose|we introduce|we define|novel approach):\s*([^.]+\.)',
        r'(?:the key is|the insight is|crucially):\s*([^.]+\.)',
        r'(?:hypothesis|conjecture):\s*([^.]+\.)',
        r'(?:observation|finding|result):\s*([^.]+\.)',
    ]

    for pattern in idea_markers:
        matches = re.findall(pattern, text, re.IGNORECASE)
        ideas.extend(matches)

    return ideas


def extract_algorithms(text):
    """Extract algorithm descriptions and pseudocode"""
    algorithms = []

    # Algorithm patterns
    algo_patterns = [
        r'(?:algorithm|procedure|function)\s+(\w+)\s*\(([^)]*)\)[^{]*\{([^}]+)\}',
        r'(?:step\s*1|1\.)\s*([^\n]+)(?:\n(?:step\s*\d|\d\.)\s*([^\n]+))+',
        r'(?:input|output):\s*([^\n]+)',
        r'(?:for|while|repeat)\s+([^\n]+)\s+(?:do|:)',
    ]

    for pattern in algo_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for m in matches:
            if isinstance(m, tuple):
                algorithms.append(' '.join(m))
            else:
                algorithms.append(m)

    return algorithms


def extract_code_blocks(text):
    """Extract code blocks"""
    code_blocks = []

    # Markdown code blocks
    pattern = r'```(\w*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for lang, code in matches:
        code_blocks.append({
            'language': lang or 'unknown',
            'code': code.strip(),
        })

    # Indented code (4 spaces)
    lines = text.split('\n')
    current_block = []
    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            current_block.append(line)
        else:
            if len(current_block) > 3:
                code_blocks.append({
                    'language': 'unknown',
                    'code': '\n'.join(current_block),
                })
            current_block = []

    return code_blocks


def identify_topics(text):
    """Identify research topics in text"""
    found_topics = []
    text_lower = text.lower()

    for topic, patterns in RESEARCH_TOPICS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_topics.append(topic)
                break

    return list(set(found_topics))


def extract_novel_terms(text):
    """Extract potentially novel terminology"""
    terms = []

    # Patterns for new term definitions
    patterns = [
        r'"([^"]+)"\s+(?:is defined as|refers to|means)',
        r'(?:we call this|we term this|named)\s+"?([^".\n]+)"?',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:is|are|refers)',  # Capitalized Terms
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        terms.extend(matches)

    return list(set(terms))


# ============================================================================
# MAIN PARSER
# ============================================================================

def parse_corpus(filepath):
    """Parse a research corpus file"""
    print(f"[*] Parsing: {filepath}")

    text = Path(filepath).read_text(errors='ignore')

    results = {
        'source': str(filepath),
        'parsed_at': datetime.now().isoformat(),
        'char_count': len(text),
        'latex': extract_latex(text),
        'equations': extract_equations(text),
        'ideas': extract_ideas(text),
        'algorithms': extract_algorithms(text),
        'code_blocks': extract_code_blocks(text),
        'topics': identify_topics(text),
        'novel_terms': extract_novel_terms(text),
    }

    # Stats
    results['stats'] = {
        'display_math_count': len(results['latex']['display_math']),
        'inline_math_count': len(results['latex']['inline_math']),
        'theorem_count': len(results['latex']['theorems']),
        'proof_count': len(results['latex']['proofs']),
        'definition_count': len(results['latex']['definitions']),
        'equation_count': len(results['equations']),
        'idea_count': len(results['ideas']),
        'algorithm_count': len(results['algorithms']),
        'code_block_count': len(results['code_blocks']),
        'topic_count': len(results['topics']),
    }

    return results


def save_results(results, output_dir=None):
    """Save parsed results"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full JSON
    json_path = output_dir / f"research_parsed_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved: {json_path}")

    # Save LaTeX equations separately
    latex_path = output_dir / f"latex_equations_{timestamp}.tex"
    with open(latex_path, 'w') as f:
        f.write("% Extracted LaTeX from Gemini Research Corpus\n")
        f.write(f"% Extracted: {results['parsed_at']}\n\n")

        f.write("% === DISPLAY MATH ===\n\n")
        for eq in results['latex']['display_math']:
            f.write(f"% Type: {eq['type']}\n")
            f.write(f"\\[\n{eq['content']}\n\\]\n\n")

        f.write("\n% === THEOREMS ===\n\n")
        for thm in results['latex']['theorems']:
            f.write(f"\\begin{{{thm['type']}}}\n{thm['content']}\n\\end{{{thm['type']}}}\n\n")

        f.write("\n% === PROOFS ===\n\n")
        for proof in results['latex']['proofs']:
            f.write(f"\\begin{{proof}}\n{proof}\n\\end{{proof}}\n\n")

        f.write("\n% === DEFINITIONS ===\n\n")
        for defn in results['latex']['definitions']:
            f.write(f"\\begin{{definition}}\n{defn}\n\\end{{definition}}\n\n")

    print(f"[+] Saved: {latex_path}")

    # Save ideas and concepts
    ideas_path = output_dir / f"research_ideas_{timestamp}.txt"
    with open(ideas_path, 'w') as f:
        f.write("RESEARCH IDEAS AND CONCEPTS\n")
        f.write("="*60 + "\n\n")

        f.write("TOPICS IDENTIFIED:\n")
        for topic in results['topics']:
            f.write(f"  • {topic}\n")

        f.write("\n\nKEY IDEAS:\n")
        for idea in results['ideas']:
            f.write(f"  → {idea}\n\n")

        f.write("\nNOVEL TERMS:\n")
        for term in results['novel_terms']:
            f.write(f"  • {term}\n")

        f.write("\n\nEQUATIONS:\n")
        for eq in results['equations'][:50]:  # First 50
            f.write(f"  {eq}\n")

    print(f"[+] Saved: {ideas_path}")

    # Save code
    if results['code_blocks']:
        code_path = output_dir / f"extracted_code_{timestamp}"
        code_path.mkdir(exist_ok=True)
        for i, block in enumerate(results['code_blocks']):
            ext = {'python': '.py', 'javascript': '.js', 'rust': '.rs'}.get(block['language'], '.txt')
            with open(code_path / f"code_{i:03d}{ext}", 'w') as f:
                f.write(block['code'])
        print(f"[+] Saved: {code_path}/")

    # Print summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    for key, val in results['stats'].items():
        print(f"  {key}: {val}")
    print(f"{'='*60}\n")

    return json_path


def watch_downloads():
    """Watch Downloads folder for new corpus files"""
    import time
    downloads = Path.home() / "Downloads"
    seen = set()

    print("[*] Watching Downloads for new corpus files...")
    print("    (Ctrl+C to stop)\n")

    while True:
        for f in downloads.glob("gemini_research_corpus*.txt"):
            if f not in seen:
                seen.add(f)
                print(f"\n[!] New corpus detected: {f.name}")
                results = parse_corpus(f)
                save_results(results)

        for f in downloads.glob("*corpus*.txt"):
            if f not in seen:
                seen.add(f)
                print(f"\n[!] New corpus detected: {f.name}")
                results = parse_corpus(f)
                save_results(results)

        time.sleep(5)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 research_parser.py <corpus_file>")
        print("       python3 research_parser.py --watch")
        return

    if sys.argv[1] == '--watch':
        watch_downloads()
    else:
        # Process provided files
        for arg in sys.argv[1:]:
            path = Path(arg).expanduser()
            if path.exists():
                results = parse_corpus(path)
                save_results(results)
            else:
                # Try glob
                for f in Path.home().glob(arg.lstrip('~/')):
                    results = parse_corpus(f)
                    save_results(results)


if __name__ == "__main__":
    main()
