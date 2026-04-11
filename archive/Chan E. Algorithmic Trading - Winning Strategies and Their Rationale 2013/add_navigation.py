import os

# Define the root directory
ROOT_DIR = "/home/restful3/workspace/ml4t/source/Chan E. Algorithmic Trading - Winning Strategies and Their Rationale 2013"

# Define the logical order of files relative to ROOT_DIR
FILES_ORDER = [
    "README.md",
    "chapter_1_backtesting_and_automated_execution/README.md",
    "chapter_2_the_basics_of_mean_reversion/README.md",
    "chapter_3_implementing_mean_reversion_strategies/README.md",
    "chapter_4_mean_reversion_of_stocks_and_etfs/README.md",
    "chapter_5_mean_reversion_of_currencies_and_futures/README.md",
    "chapter_6_interday_momentum_strategies/README.md",
    "chapter_7_intraday_momentum_strategies/README.md",
    "chapter_8_risk_management/README.md",
    "conclusion/README.md"
]

def get_relative_path(from_file, to_file):
    """Calculates the relative path from one file to another."""
    from_dir = os.path.dirname(os.path.join(ROOT_DIR, from_file))
    to_path = os.path.join(ROOT_DIR, to_file)
    return os.path.relpath(to_path, from_dir)

def add_navigation():
    for i, file_rel_path in enumerate(FILES_ORDER):
        file_path = os.path.join(ROOT_DIR, file_rel_path)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        # Determine Previous, Next, and TOC paths
        prev_link = None
        next_link = None
        toc_link = get_relative_path(file_rel_path, "README.md")
        
        if i > 0:
            prev_rel = FILES_ORDER[i-1]
            prev_path = get_relative_path(file_rel_path, prev_rel)
            prev_link = f"[< Previous]({prev_path})"
        else:
            prev_link = "Previous" # Disabled text or empty

        if i < len(FILES_ORDER) - 1:
            next_rel = FILES_ORDER[i+1]
            next_path = get_relative_path(file_rel_path, next_rel)
            next_link = f"[Next >]({next_path})"
        else:
            next_link = "Next" # Disabled text

        # Construct the navigation footer
        # Using a distinct separator to easily identify and replace if needed later
        nav_content = f"\n\n---\n\n<div align=\"center\">\n\n{prev_link} | [Table of Contents]({toc_link}) | {next_link}\n\n</div>"
        
        # Read content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if navigation already exists (simple check for now)
        if '<div align="center">' in content and '[Table of Contents]' in content:
             # Remove old navigation if it looks like ours (simplified update mechanism)
             # For this task, we assume we append. If we re-run, we might duplicate unless we check.
             # Let's clean up previous runs if detected strictly at the end.
             lines = content.splitlines()
             if lines[-1].strip().endswith("</div>") and "[Table of Contents]" in lines[-1]:
                 # Try to strip the last few lines if they match our pattern
                 # This is a basic heuristic.
                 pass

        # Since the user asked to ADD, I will append. 
        # But to avoid infinite appending on re-runs, let's check if the exact string exists at the end?
        # A safer bet for a script is to read, strip trailing whitespace, and append.
        
        # NOTE: If we want to be idempotent, we should check if the footer is already there.
        # I'll strip any existing matching footer to be safe, then append new one.
        
        lines = content.splitlines()
        # Look for the footer signature
        # We expect: ---, empty, <div align="center">...
        
        # Let's just append for now as requested, but maybe add a check
        if "[Table of Contents]" in content[-200:]: # Check last 200 chars roughly
             print(f"Skipping {file_rel_path} (Navigation seems to verify exist)")
             # Actually, let's force update it to be sure it's correct
             # Find the start of the footer
             if "\n---\n\n<div align=\"center\">" in content:
                 content = content.split("\n---\n\n<div align=\"center\">")[0]
             
        
        new_content = content.rstrip() + nav_content + "\n"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print(f"Updated {file_rel_path}")

if __name__ == "__main__":
    add_navigation()
