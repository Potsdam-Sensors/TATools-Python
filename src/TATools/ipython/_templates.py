import re
import ast

# {{name|default}} — default parsed as a Python literal (int/float/bool/list/dict/str)
_PATTERN = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*(?:\|\s*(.*?)\s*)?\}\}")

def parse_literal(text: str):
    """Safely parse Python literals: 10, 3.14, True, [1,2], {'a':1}, 'hi'."""
    if text is None:
        return None
    try:
        return ast.literal_eval(text.strip())
    except Exception:
        # Fallback: treat as raw token; for strings, require quotes in templates/params.
        return text

def render_template(text: str, params: dict) -> str:
    """Substitute {{key|default}} with params[key] if present, else default (parsed)."""
    def repl(m):
        key, default_txt = m.group(1), m.group(2)
        if key in params:
            val = params[key]
        elif default_txt is not None:
            val = parse_literal(default_txt)
        else:
            return m.group(0)  # leave untouched
        return str(val)
    return _PATTERN.sub(repl, text)
