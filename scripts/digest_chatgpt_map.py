#!/usr/bin/env python3
"""
Digest ChatGPT conversation export and compare to current master architecture.

This script extracts all 'decisions' and 'projects' bullet points from a ChatGPT JSON export,
deduplicates them, and compares them against the master architecture defined in
master_project_analyzer.create_master_architecture(). It then generates a Markdown report
highlighting unique decisions, projects, current architecture items, and differences.
"""
import os
import sys
import re
import json
import argparse

# Ensure the repository root is on the import path for master_project_analyzer
_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir)))

from master_project_analyzer import create_master_architecture


def extract_bullets(text):
    """
    Extract bullet list items from a block of text.
    Supports '-', '*', and numbered list prefixes.
    """
    items = []
    for line in text.splitlines():
        stripped = line.strip()
        # match bullet points: '-', '*', or numbered lists '1. '
        if stripped.startswith('- ') or stripped.startswith('* ') or re.match(r'^\d+\.\s+', stripped):
            # remove leading bullet markers and whitespace
            item = re.sub(r'^[-*]\s+|^\d+\.\s+', '', stripped)
            items.append(item)
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Digest ChatGPT export and compare decisions/projects against master architecture"
    )
    parser.add_argument(
        '--json-path',
        default=os.path.expanduser('~/Downloads/conversations.json'),
        help='Path to ChatGPT export JSON file'
    )
    parser.add_argument(
        '--output',
        help='Path to write Markdown report (defaults to stdout)'
    )
    args = parser.parse_args()

    # Load ChatGPT conversation export
    with open(args.json_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    decision_items = set()
    project_items = set()

    # Iterate through each conversation and extract relevant bullet points
    for convo in conversations:
        mapping = convo.get('mapping', {}) or {}
        for node in mapping.values():
            msg = node.get('message')
            if not msg or not msg.get('content'):
                continue
            parts = msg['content'].get('parts') or []
            for part in parts:
                # only process text parts
                if not isinstance(part, str):
                    continue
                bullets = extract_bullets(part)
                for item in bullets:
                    low = item.lower()
                    if 'decision' in low:
                        decision_items.add(item)
                    if 'project' in low:
                        project_items.add(item)

    # Load current master architecture
    architecture = create_master_architecture()
    arch_items = set()

    def _recurse(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _recurse(v)
        elif isinstance(obj, list):
            for v in obj:
                if isinstance(v, str):
                    arch_items.add(v)
                else:
                    _recurse(v)

    _recurse(architecture)

    # Build report
    lines = []
    lines.append('# ChatGPT Conversations Decisions and Projects Digest')
    lines.append('')
    lines.append('## Unique Decisions Found')
    if decision_items:
        for d in sorted(decision_items):
            lines.append(f'- {d}')
    else:
        lines.append('- (none)')
    lines.append('')
    lines.append('## Unique Projects Found')
    if project_items:
        for p in sorted(project_items):
            lines.append(f'- {p}')
    else:
        lines.append('- (none)')
    lines.append('')
    lines.append('## Current Architecture Items')
    if arch_items:
        for a in sorted(arch_items):
            lines.append(f'- {a}')
    else:
        lines.append('- (none)')
    lines.append('')
    lines.append('## Items in ChatGPT but Missing in Architecture')
    chat_set = decision_items.union(project_items)
    missing = sorted(chat_set - arch_items)
    if missing:
        for m in missing:
            lines.append(f'- {m}')
    else:
        lines.append('- None 🎉')
    lines.append('')
    lines.append('## Items in Architecture but not in ChatGPT')
    extra = sorted(arch_items - chat_set)
    if extra:
        for e in extra:
            lines.append(f'- {e}')
    else:
        lines.append('- None 🎉')

    report = '\n'.join(lines)

    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as out:
            out.write(report)
        print(f'Report written to {args.output}')
    else:
        print(report)


if __name__ == '__main__':
    main()