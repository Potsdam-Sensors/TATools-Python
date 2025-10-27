# Minimal IPython extension with magics for Jupyter/VS Code

from IPython.core.magic import register_line_magic, register_cell_magic
from IPython.display import display, Markdown
from IPython import get_ipython

from ._examples import list_examples, load_example_text
from ._templates import render_template, parse_literal


def _insert_next_cell(code: str) -> None:
    ip = get_ipython()
    if ip is None:
        raise RuntimeError("This magic must run inside IPython/Jupyter.")
    ip.set_next_input(code, replace=False)


@register_line_magic
def tat_examples(line: str = ""):
    """
    %tat_examples
    List available example snippet names.
    """
    ex = list_examples()
    if not ex:
        display(Markdown("_No examples found._"))
        return
    bullets = "\n".join(f"- `{name}` — {meta.get('desc','')}" for name, meta in ex.items())
    display(Markdown(f"**TATools examples**\nGenerate template code.\n\nTry a cell with `%tag_example <example-name-from-below>`.\n\n{bullets}"))



@register_line_magic
def tat_example(line: str):
    parts = [p for p in line.split() if p.strip()]
    if not parts:
        raise ValueError("Usage: %tat_example <name> [key=value ...]")
    name, *kv = parts

    # parse key=value into typed params (numbers/bools/lists allowed)
    params = {}
    for item in kv:
        if "=" in item:
            k, v = item.split("=", 1)
            params[k] = parse_literal(v)

    code = load_example_text(name)      # your existing loader
    code = render_template(code, params)
    _insert_next_cell(code)
    display(Markdown(f"✅ Inserted example `{name}`."))



@register_cell_magic
def tat_template(line: str, cell: str):
    """
    %%tat_template key1=val key2=val
    Render the cell as a template and insert the result into the next cell.
    """
    params = {}
    for item in (line or "").split():
        if "=" in item:
            k, v = item.split("=", 1)
            params[k] = v
    out = render_template(cell, params)
    _insert_next_cell(out)
    display(Markdown("✅ Inserted rendered template into the next cell."))


def load_ipython_extension(ip):
    # Called by: %load_ext TATools.ipython
    # (decorators already registered on import; nothing else needed)
    pass


def unload_ipython_extension(ip):
    # Optional: nothing to clean up for simple magics
    pass
