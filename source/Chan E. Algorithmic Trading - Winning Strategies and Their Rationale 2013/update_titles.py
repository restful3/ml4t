import os
import re

# Define the root directory
ROOT_DIR = "/home/restful3/workspace/ml4t/source/Chan E. Algorithmic Trading - Winning Strategies and Their Rationale 2013"

def to_title_case(text):
    # Split by underscores or spaces and capitalize
    words = text.replace('_', ' ').split()
    # Capitalize first letter of each word
    return ' '.join(word.capitalize() for word in words)

def update_titles():
    # Iterate through immediate subdirectories
    for item in os.listdir(ROOT_DIR):
        item_path = os.path.join(ROOT_DIR, item)
        if not os.path.isdir(item_path):
            continue

        new_title = ""
        
        # Match pattern: chapter_1_name
        match = re.match(r"^chapter_(\d+)_(.+)$", item)
        if match:
            chapter_num = int(match.group(1))
            chapter_name = match.group(2)
            # Format: # ch01 Backtesting And Automated Execution
            new_title = f"# ch{chapter_num:02d} {to_title_case(chapter_name)}"
        elif item == "conclusion":
            new_title = "# Conclusion"
        else:
            # Skip folders that don't match the pattern (unless we want to handle them differently)
            print(f"Skipping folder: {item} (pattern mismatch)")
            continue

        readme_path = os.path.join(item_path, "README.md")
        if not os.path.exists(readme_path):
            print(f"Warning: No README found in {item}")
            continue

        with open(readme_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Warning: Empty README in {item}")
            continue

        # Check if first line needs update
        current_first_line = lines[0].strip()
        if current_first_line != new_title:
            print(f"Updating {item}/README.md title:")
            print(f"  Old: {current_first_line}")
            print(f"  New: {new_title}")
            
            lines[0] = new_title + "\n"
            
            with open(readme_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        else:
            print(f"Skipping {item} (Title already matches)")

if __name__ == "__main__":
    update_titles()
