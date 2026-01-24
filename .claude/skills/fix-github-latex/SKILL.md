---
name: fix-github-latex
description: Fixes LaTeX formulas in Markdown files to be compatible with GitHub rendering, avoiding common pitfalls like \tag{} and incorrect italicization.
---

# Fix GitHub LaTeX Compatibility

This skill guides the AI to correct mathematical formulas in Markdown files so they render correctly on GitHub. GitHub's markdown renderer often fails with standard spacing command `\tag` and interprets plain text in math blocks as variables.

## Core Rules

1.  **NO `\tag{}`**: GitHub's math renderer often breaks layout or renders vertically when `\tag` is used.
    *   **Incorrect**: `$$ f(x) = x^2 \tag{1.1} $$`
    *   **Correct**: `$$ f(x) = x^2 \qquad (1.1) $$`
    *   **Action**: Replace `\tag{...}` with manual spacing `\qquad (...)` at the end of the line.

2.  **Use `\text{}` for Function Names**: Avoid "italicized" function names which look like variables multiplied together.
    *   **Incorrect**: `mean(x)`, `std(x)`, `sign(i)` (Renders as $m \times e \times a \times n$)
    *   **Correct**: `\text{mean}(x)`, `\text{std}(x)`, `\text{sign}(i)`
    *   **Action**: Wrap standard function names and descriptive text in `\text{}`.

3.  **Inline Math using `$`**: Do not use Markdown italics for variables.
    *   **Incorrect**: The variable *x* and function *f*...
    *   **Correct**: The variable $x$ and function $f$...
    *   **Action**: Convert `*variable*` or `_variable_` to `$variable$`.

4.  **Use `\frac{}{}`**: Prefer LaTeX fraction syntax over `/` for arithmetic expressions in display math.
    *   **Incorrect**: `(a + b) / c`
    *   **Correct**: `\frac{a + b}{c}`

## Execution Steps

1.  **Scan**: Read the Markdown file and identify both Display Math (`$$...$$`) and Inline Math (`$...$`).
2.  **Identify**: Look for `\tag`, text-like variables without `\text`, and markdown italics acting as math variables.
3.  **Refactor**: Apply the rules above to rewrite the formulas.
4.  **Verify**: Ensure the mathematical meaning is unchanged and the formatting is clean.
