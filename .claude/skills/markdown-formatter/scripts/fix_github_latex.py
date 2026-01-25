#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix LaTeX formulas in Markdown files for GitHub compatibility.
"""

import re
import sys

def fix_latex(content: str) -> str:
    """
    Fixes LaTeX formulas in Markdown content.
    """
    
    # 1. Replace \tag{...} with \qquad (...) in display math
    # Pattern: Look for $$ ... \tag{X} ... $$
    # This is complex because $$ can be multi-line.
    # We'll use a regex that matches content between $$ ... $$
    
    def replace_tag(match):
        math_content = match.group(1)
        # Check for \tag{...}
        # We want to replace \tag{something} with \qquad (something)
        # And ensure it's at the end of the line if possible, or just append it.
        
        def tag_replacer(tag_match):
            tag_content = tag_match.group(1)
            return f"\\qquad ({tag_content})"
        
        new_math_content = re.sub(r'\\tag\{([^}]+)\}', tag_replacer, math_content)
        return f"$${new_math_content}$$"

    # Match display math blocks $$ ... $$ (dot matches newline)
    content = re.sub(r'\$\$(.+?)\$\$', replace_tag, content, flags=re.DOTALL)

    # 2. Wrap standard function names in \text{} if not already
    funcs = ["mean", "std", "var", "sign", "log", "exp", "sin", "cos", "tan", "min", "max", "arg(?:min|max)"]
    
    def apply_text_macro(math_text):
        for func in funcs:
            # Pattern: look for function name not preceded by \ or { or (letter) and not followed by letter
            # Exclude if preceded by { to avoid \text{mean} -> \text{\text{mean}}
            regex = r'(?<!\\)(?<!\{)(?<![a-zA-Z])(' + func + r')(?![a-zA-Z])'
            math_text = re.sub(regex, r'\\text{\1}', math_text)
        return math_text

    def process_display_math(match):
        inner = match.group(1)
        fixed_inner = apply_text_macro(inner)
        return f"$${fixed_inner}$$"

    def process_inline_math(match):
        inner = match.group(1)
        fixed_inner = apply_text_macro(inner)
        return f"${fixed_inner}$"

    # Apply to $$...$$
    content = re.sub(r'\$\$(.+?)\$\$', process_display_math, content, flags=re.DOTALL)

    # Apply to $...$
    # Regex must ensure it doesn't match inside $$...$$
    # Use negative lookbehind/lookahead to ensure single $
    content = re.sub(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)', process_inline_math, content)
    
    # 3. Convert *x* to $x$ for single variables
    # This is also context sensitive.
    # Pattern: *[a-zA-Z]* or _[a-zA-Z]_ surrounding a single letter or short variable?
    # Let's assume single letter variables for now as per SKILL.md rules
    
    # Replace *x* -> $x$
    content = re.sub(r' (?:\*|_)([a-zA-Z])(?:\*|_) ', r' $\1$ ', content)
    # Start of line
    content = re.sub(r'^(?:\*|_)([a-zA-Z])(?:\*|_) ', r'$\1$ ', content, flags=re.MULTILINE)
    # End of line
    content = re.sub(r' (?:\*|_)([a-zA-Z])(?:\*|_)[.?,]', r' $\1$', content)
    
    return content

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fix_github_latex.py <filepath>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        fixed_content = fix_latex(content)

        if content != fixed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✓ LaTeX Fixed: {file_path}")
        else:
            print(f"○ No LaTeX changes needed: {file_path}")

    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
