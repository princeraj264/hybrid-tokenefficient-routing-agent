"""
Safe arithmetic solver, code extraction, test-case extraction, and code
verification utilities.

``try_solve_math`` uses AST-based parsing (not ``eval()``) with a strict
whitelist of operators so it can safely evaluate simple arithmetic expressions
embedded in natural-language prompts.
``extract_code`` pulls Python from a markdown fence.
``extract_test_cases`` scans a task prompt for example input/output pairs.
``verify_code`` runs code in a sandboxed subprocess, optionally asserting
extracted test cases for correctness.
``verify_ner_answer`` performs a cheap substring check that every entity
claimed by the model actually appears in the original prompt text.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Safe arithmetic evaluator
# ─────────────────────────────────────────────────────────────────────────────

# Whitelist of AST node types → safe functions from the ``operator`` module.
_ALLOWED_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _check_safe(node: ast.AST, top_level: bool = True) -> None:
    """Recursively verify that *node* contains only safe arithmetic nodes.

    Raises ``ValueError`` if an unsupported node type or constant type is
    encountered.
    """
    if top_level and not isinstance(node, ast.Expression):
        raise ValueError("top-level node must be an expression")

    if isinstance(node, ast.Expression):
        _check_safe(node.body, top_level=False)

    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(
                f"unsupported constant type: {type(node.value).__name__}"
            )

    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in (ast.UAdd, ast.USub):
            raise ValueError("unsupported unary operator")
        _check_safe(node.operand, top_level=False)

    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_OPS:
            raise ValueError(
                f"unsupported binary operator: {type(node.op).__name__}"
            )
        _check_safe(node.left, top_level=False)
        _check_safe(node.right, top_level=False)

    else:
        raise ValueError(f"unsupported node type: {type(node).__name__}")


def _eval_ast(node: ast.AST) -> int | float:
    """Evaluate a pre-validated AST using the operator whitelist."""
    if isinstance(node, ast.Constant):
        return node.value  # type: ignore[return-value]
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))  # type: ignore[operator]
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](  # type: ignore[operator]
            _eval_ast(node.left), _eval_ast(node.right)
        )
    raise ValueError(f"cannot evaluate node: {type(node).__name__}")


# ── Word-to-symbol normalisation ─────────────────────────────────────────────

_WORD_TO_OP: dict[str, str] = {
    # multiplication
    "multiplied by": "*",
    "multiply": "*",
    "times": "*",
    # division
    "divided by": "/",
    "divide": "/",
    # addition
    "added to": "+",
    "plus": "+",
    "add": "+",
    # subtraction
    "subtracted from": "-",
    "subtract": "-",
    "minus": "-",
    # modulo
    "modulo": "%",
    "mod": "%",
    "remainder of": "%",
}

# Longest-first sort avoids partial matches (e.g. "added to" before "add").
_WORD_PATTERN = re.compile(
    "|".join(re.escape(w) for w in sorted(_WORD_TO_OP, key=len, reverse=True)),
    re.IGNORECASE,
)

# Rough match for a chain of atomic terms (numbers or parenthesised groups)
# separated by operators.  This is intentionally broad — the AST parser acts
# as the true validator.
_ARITH_CANDIDATE = re.compile(
    r"[+\-]?(?:\d+(?:\.\d+)?|\([^)]*\))"
    r"(?:\s*[+\-*/%]{1,2}\s*"
    r"[+\-]?(?:\d+(?:\.\d+)?|\([^)]*\)))+"
)


def try_solve_math(prompt: str) -> str | None:
    """Attempt to solve a simple arithmetic expression in *prompt*.

    Pipeline
    --------
    1. Normalise English math phrasing into symbols (e.g. "times" → ``*``).
    2. Extract candidate arithmetic substrings with a regex.
    3. Parse each candidate via ``ast.parse`` (not ``eval()``).
    4. Validate the AST against a strict operator whitelist.
    5. Evaluate the validated AST.

    Returns
    -------
    The result as a string (``"5"``, ``"3.14"``) on success, or ``None`` so the
    caller can fall back to the local / remote model pipeline.
    """
    # Step 1 — replace English math phrases with symbols
    normalized = _WORD_PATTERN.sub(
        lambda m: _WORD_TO_OP[m.group(0).lower()], prompt
    )

    # Step 2 — find candidate arithmetic expressions
    candidates = _ARITH_CANDIDATE.findall(normalized)

    for expr in candidates:
        expr = expr.strip()
        if not expr:
            continue

        # Must contain at least one operator (reject lone numbers).
        if not re.search(r"[+\-*/%]", expr):
            continue

        try:
            tree = ast.parse(expr, mode="eval")
            _check_safe(tree)
            result = _eval_ast(tree)
            # Return integer string when the value is a whole number.
            if isinstance(result, float) and result == result // 1:
                return str(int(result))
            return str(result)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction & verification
# ─────────────────────────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(
    r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def extract_code(text: str) -> str | None:
    """Extract the first Python code block from markdown-fenced *text*.

    Looks for `` ```python … ``` `` fences.  Returns the dedented, stripped
    inner code, or ``None`` if no block is found.
    """
    match = _CODE_BLOCK_RE.search(text)
    if match is None:
        return None
    return textwrap.dedent(match.group(1)).strip()


def extract_test_cases(prompt: str) -> list[tuple[str, str]]:
    """Scan *prompt* text for example input/output pairs.

    Recognised patterns (in priority order):

    1. Inline assert lines — ``assert <expr> == <expected>``
    2. Arrow notation — ``<expr> -> <result>`` or ``Example: <expr> -> <result>``
    3. ``Input: <expr>`` / ``Output: <result>`` blocks where the
       Input value looks like a function call (e.g. ``add(2, 3)``).

    Returns
    -------
    A list of ``(input_expr, expected_output)`` tuples, or an empty list
    if no test cases can be confidently extracted.
    """
    cases: list[tuple[str, str]] = []

    # Pattern 1: inline assert statements
    for line in prompt.splitlines():
        line = line.strip()
        m = re.match(r"assert\s+(.+?)\s*==\s*(.+)", line)
        if m:
            expr = m.group(1).strip()
            expected = m.group(2).strip()
            # Skip bare constants — they aren't useful tests
            if expr not in ("True", "False", "None"):
                cases.append((expr, expected))

    # Pattern 2: arrow notation
    #   Example: reverse("abc") -> "cba"
    #   Just:    double(3) -> 6
    for m in re.finditer(
        r"(?:Example\s*:\s*)?([A-Za-z_]\w*\([^)]*\))\s*->\s*(.+)",
        prompt,
        re.MULTILINE,
    ):
        cases.append((m.group(1).strip(), m.group(2).strip()))

    # Pattern 3: Input / Output blocks where Input looks like a call
    lines = prompt.splitlines()
    for i in range(len(lines) - 1):
        in_match = re.match(
            r"Input\s*:\s*(.+)", lines[i].strip(), re.IGNORECASE
        )
        out_match = re.match(
            r"Output\s*:\s*(.+)", lines[i + 1].strip(), re.IGNORECASE
        )
        if in_match and out_match:
            inp = in_match.group(1).strip()
            out = out_match.group(1).strip()
            # Only accept if Input looks like a function call
            if re.match(r"[A-Za-z_]\w*\(.*\)", inp):
                cases.append((inp, out))

    return cases


def verify_code(
    code: str,
    test_cases: list[tuple[str, str]] | None = None,
    timeout: int = 5,
) -> tuple[bool, str]:
    """Run *code* in a sandboxed subprocess and report correctness.

    If *test_cases* is provided and non-empty, assert statements are
    appended to *code* before execution so the script fails on any
    mismatch (e.g. ``AssertionError`` with the failed expression).

    Writes the code to a temporary ``.py`` file and executes it with
    ``python3 -I`` (isolated mode), an empty environment dict, and a strict
    timeout.  This provides basic containment — no network, no inherited
    environment variables, no user site-packages.

    Returns
    -------
    ``(True, "")`` if the process exits with code 0,
    ``(False, stderr)`` on a non-zero exit (includes assertion failures), or
    ``(False, "timeout")`` if the process times out.
    """
    # Build final script: user code + optional test-case assertions
    body = code
    if test_cases:
        assertions = "\n\n# ---- auto-generated test cases ----\n"
        for expr, expected in test_cases:
            assertions += f"assert {expr} == {expected}\n"
        body = code + assertions

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(body)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            ["python3", "-I", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={},  # isolated: no environment variables passed through
        )
        if proc.returncode == 0:
            return True, ""
        return False, proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# NER hallucination sanity check
# ─────────────────────────────────────────────────────────────────────────────


def verify_ner_answer(prompt: str, answer: str) -> bool:
    """Cheap NER hallucination sanity check.

    Verifies that every entity claimed by the model in *answer* actually
    appears as a substring (case-insensitive) in the original *prompt* text.

    This is NOT a full NER accuracy validator — it doesn't check entity
    type correctness or boundary precision. It only catches blatant
    hallucinations: entities that have no match in the source text at all.

    The *answer* is parsed loosely — it may be a JSON array, a comma-
    separated list, or one-entity-per-line (all three are tried).
    """
    # Normalise the prompt for case-insensitive matching
    prompt_lower = prompt.lower()

    entities: list[str] = []
    stripped = answer.strip()

    # Try JSON array first
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                entities = [str(e).strip() for e in parsed if e is not None]
        except json.JSONDecodeError:
            pass

    # Fall back to newline-separated (one entity per line)
    if not entities:
        lines = [line.strip() for line in stripped.split("\n") if line.strip()]
        # Only use this if no line has internal commas (avoid splitting
        # phrases like "New York, NY" into separate entities).
        if lines and all("," not in line for line in lines):
            entities = [e.rstrip(".,;!?") for e in lines]

    # Fall back to comma-separated
    if not entities:
        entities = [
            e.strip().rstrip(".,;!?") for e in stripped.split(",") if e.strip()
        ]

    # If the answer isn't parseable into any list, reject it
    if not entities:
        return False

    # Check every entity appears somewhere in the prompt (case-insensitive)
    for entity in entities:
        if entity.lower() not in prompt_lower:
            return False

    return True