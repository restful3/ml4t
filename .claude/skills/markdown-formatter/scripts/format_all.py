#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master script to apply all Markdown formatting rules (Bold + LaTeX).
"""

import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 format_all.py <filepath>")
        sys.exit(1)

    file_path = sys.argv[1]
    
    # Get the directory of this script to find the others
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    bold_script = os.path.join(script_dir, "fix_multilingual_bold.py")
    latex_script = os.path.join(script_dir, "fix_github_latex.py")
    code_blocks_script = os.path.join(script_dir, "fix_code_blocks.py")
    
    print(f"Formating file: {file_path}")
    
    # Run Bold Fixer
    print(">>> Running Multilingual Bold Fixer...")
    try:
        subprocess.run([sys.executable, bold_script, file_path], check=True)
    except subprocess.CalledProcessError:
        print("Bold fixer failed.")
        
    # Run LaTeX Fixer
    print(">>> Running GitHub LaTeX Fixer...")
    try:
        subprocess.run([sys.executable, latex_script, file_path], check=True)
    except subprocess.CalledProcessError:
        print("LaTeX fixer failed.")
        
    # Run Code Block Fixer
    print(">>> Running Code Block Fixer...")
    try:
        subprocess.run([sys.executable, code_blocks_script, file_path], check=True)
    except subprocess.CalledProcessError:
        print("Code block fixer failed.")

    print("Formatting complete.")

if __name__ == '__main__':
    main()
