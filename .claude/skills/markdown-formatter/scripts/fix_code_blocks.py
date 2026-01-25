#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix missing language identifiers in Markdown code blocks.
Auto-detects Python and Bash.
"""

import sys
import re

def detect_language(code_content):
    """
    Heuristically detect if the code block is Python or Bash.
    Returns 'python', 'bash', or None.
    """
    # Python heuristics
    python_keywords = [
        r'\bdef\s+', r'\bclass\s+', r'\bimport\s+', r'\bfrom\s+.*\s+import\s+',
        r'print\(', r'if\s+__name__\s*==\s*[\'"]__main__[\'"]:',
        r'\breturn\s+', r'\braise\s+', r'\btry:', r'\bexcept\s+.*:',
        r'\bwith\s+open\(', r'\s*#\s*.*coding:\s*utf-8'
    ]
    
    # Bash heuristics
    bash_keywords = [
        r'^\s*\$\s+', r'\bsudo\s+', r'\bnpm\s+', r'\bpip\s+', r'\bapt-get\s+',
        r'\becho\s+', r'\bls\s+', r'\bcd\s+', r'\bmkdir\s+', r'\brm\s+',
        r'\bcp\s+', r'\bmv\s+', r'\bgit\s+', r'\bexport\s+', r'\bsource\s+',
        r'\bchmod\s+', r'\bchown\s+', r'\bcat\s+', r'\bgrep\s+', r'\.\/'
    ]

    # MATLAB codes
    matlab_keywords = [
        r'%\s+', r'\bend\s*$', r'\bfunction\s+', r'\.\.\.\s*$',
        r'\bdisp\(', r'\bfprintf\(', r'\bzeros\(', r'\bones\(',
        r'\bNaN\(', r'\bsize\(', r'\blength\(', r'\bfind\(',
        r'\bplot\(', r'\btitle\(', r'\blegend\(', r'\bsubplot\(',
        r'\bgrid\s+on', r'\bhold\s+on', r'\bdetectImportOptions',
        r'\breadtable'
    ]

    # Check for Python
    python_score = 0
    for keyword in python_keywords:
        if re.search(keyword, code_content, re.MULTILINE):
            python_score += 1
            
    # Check for Bash
    bash_score = 0
    for keyword in bash_keywords:
        if re.search(keyword, code_content, re.MULTILINE):
            bash_score += 1

    # Check for MATLAB
    matlab_score = 0
    for keyword in matlab_keywords:
        if re.search(keyword, code_content, re.MULTILINE):
            matlab_score += 1

    if python_score > 0 and python_score >= bash_score and python_score >= matlab_score:
        return 'python'
    elif bash_score > 0 and bash_score >= python_score and bash_score >= matlab_score:
        return 'bash'
    elif matlab_score > 0 and matlab_score > python_score and matlab_score > bash_score:
        return 'matlab'
    
    return None

def fix_code_blocks(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

    new_lines = []
    in_code_block = False
    code_block_content = []
    code_block_start_index = -1
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        
        # Check for code block start/end
        if stripped_line.startswith('```'):
            # If it's a start of a block (and not end of one)
            if not in_code_block:
                # Check if it already has a language
                lang_match = re.match(r'^```(\w+)', stripped_line)
                if lang_match:
                    # Already has language, just keep it
                    new_lines.append(line)
                    in_code_block = True
                else:
                    # No language, might need detection
                    in_code_block = True
                    code_block_start_index = len(new_lines)
                    new_lines.append(line) # Add placeholder, will update later if needed
                    code_block_content = []
            else:
                # End of code block
                in_code_block = False
                
                # If we were tracking a block without language, detect now
                if code_block_start_index != -1:
                    content_str = "".join(code_block_content)
                    detected = detect_language(content_str)
                    
                    if detected:
                        # Update the start line
                        new_lines[code_block_start_index] = f"```{detected}\n"
                        print(f"  -> Detected {detected} code block at line {code_block_start_index + 1}")
                    
                    code_block_start_index = -1
                    code_block_content = []
                
                new_lines.append(line)
        else:
            if in_code_block and code_block_start_index != -1:
                code_block_content.append(line)
            new_lines.append(line)
        
        i += 1

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Processed code blocks in {file_path}")
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 fix_code_blocks.py <filepath>")
        sys.exit(1)
    
    fix_code_blocks(sys.argv[1])
