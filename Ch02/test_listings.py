"""Maintainer check: the code here matches the chapter's listings.

This is not part of working through the chapter. It reads the chapter manuscript,
pulls out every listing, and compares it with the corresponding code in this
folder, so that a listing and the code it describes cannot drift apart.

Python listings are compared as syntax trees (comments, annotation markers such as
`#A`, docstrings and formatting do not matter; every name does). The CUDA listing is
compared token by token with matmulkernel.cu.

    MDPP_CH02_MANUSCRIPT=/path/to/manuscript.md python -m pytest test_listings.py

Skips cleanly when the manuscript is not available.
"""

from __future__ import annotations

import ast
import inspect
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "..", "mdpp-book", "chapters", "ch00-demm-lite-block-matmul",
                       "manuscript", "ch00-draft-v21.md")
MANUSCRIPT = os.environ.get("MDPP_CH02_MANUSCRIPT", DEFAULT)

if not os.path.exists(MANUSCRIPT):
    pytest.skip("chapter manuscript not available", allow_module_level=True)

TEXT = open(MANUSCRIPT, encoding="utf-8").read()

# Documented deviations: statements the repo adds at the top of a listing's body.
ADDED_LEADING_STATEMENTS = {}


# ------------------------------------------------------------ extraction

def numbered_listings():
    """{'2.6': ('python', code), ...} for every '**Listing 2.N ...**' with a code block."""
    out = {}
    for m in re.finditer(r"\*\*Listing (2\.\d+)[^\n]*\*\*\s*\n+```(\w+)\n(.*?)```", TEXT, re.S):
        out[m.group(1)] = (m.group(2), m.group(3))
    return out


def python_blocks():
    return re.findall(r"```python\n(.*?)```", TEXT, re.S)


def parse(code):
    return ast.parse(code)


def function_in_text(name):
    for block in python_blocks():
        if f"def {name}(" in block:
            for node in parse(block).body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return node
    raise AssertionError(f"no listing defines {name}")


def function_in_repo(module, name):
    tree = parse(inspect.getsource(module))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{module.__name__} defines no {name}")


def body(node, skip_leading=0, drop_return=False):
    stmts = list(node.body)
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(getattr(stmts[0], "value", None), ast.Constant) \
            and isinstance(stmts[0].value.value, str):
        stmts = stmts[1:]                                   # docstring
    stmts = stmts[skip_leading:]
    if drop_return and stmts and isinstance(stmts[-1], ast.Return):
        stmts = stmts[:-1]
    return stmts


def dump(stmts):
    return [ast.dump(s, include_attributes=False) for s in stmts]


def same_function(text_node, repo_node, skip_leading=0):
    a = dump(body(text_node))
    b = dump(body(repo_node, skip_leading))
    args_a = ast.dump(text_node.args, include_attributes=False)
    args_b = ast.dump(repo_node.args, include_attributes=False)
    return a == b and _names(text_node.args) == _names(repo_node.args), (a, b, args_a, args_b)


def _names(args):
    return [a.arg for a in args.args]


def program_core(code, parameters):
    """A listing written as a module-level program, minus its setup.

    Drops imports, reading the kernel source into _SRC, and the assignments that
    the repository turns into function parameters.
    """
    stmts = []
    for node in parse(code).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in parameters | {"_HERE"}:
            continue
        if isinstance(node, ast.With) and "_SRC" in ast.dump(node) and "open" in ast.dump(node):
            continue
        stmts.append(node)
    return stmts


# ------------------------------------------------------------ Python listings

@pytest.mark.parametrize("name, module_name", [
    ("matmul_launcher", "matmul_launcher"),       # listing 2.6
    ("matrix_mul", "matmul_launcher"),            # section 2.3.2
    ("orth", "demm_launcher"),                    # section 2.3.2
    ("demm_launcher", "demm_launcher"),           # listing 2.8
])
def test_function_listing_matches_repo(name, module_name):
    module = __import__(module_name)
    ok, detail = same_function(function_in_text(name), function_in_repo(module, name),
                               ADDED_LEADING_STATEMENTS.get(name, 0))
    assert ok, f"{name}: listing and repository differ"


def test_listing_2_7_parses():
    parse(numbered_listings()["2.7"][1])


def test_listing_2_7_matches_repo():
    code = numbered_listings()["2.7"][1]
    text_core = program_core(code, {"device_ids", "b_m", "b_n", "m", "p", "n"})
    import blocked_matmul_coordinator as mod
    repo = body(function_in_repo(mod, "blocked_matmul"), drop_return=True)
    repo = [s for s in repo if not (isinstance(s, ast.Assign) and s.targets[0].id in {"m", "p", "n"})]
    assert dump(text_core) == dump(repo)


def test_listing_2_9_matches_repo():
    import demm_coordinator as mod
    text_core = program_core(numbered_listings()["2.9"][1],
                             {"device_ids", "q", "s", "h", "m", "p", "n"})
    repo = body(function_in_repo(mod, "demm_matmul"), drop_return=True)
    repo = [s for s in repo if not (isinstance(s, ast.Assign) and s.targets[0].id in {"m", "p", "n"})]
    assert dump(text_core) == dump(repo)


def test_listing_2_9_settings_are_the_reported_ones():
    """Listing 2.9's q, s and h are the defaults here and the harness's settings,
    so running the listing as printed reproduces Table 2.1."""
    import json
    import demm_coordinator
    import demm_gram_coordinator
    listed = {}
    for node in parse(numbered_listings()["2.9"][1]).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)                 and node.targets[0].id in {"q", "s", "h"}:
            listed[node.targets[0].id] = ast.literal_eval(node.value)
    for fn in (demm_coordinator.demm_matmul, demm_gram_coordinator.demm_gram):
        defaults = {k: v.default for k, v in inspect.signature(fn).parameters.items()
                    if k in listed}
        assert defaults == listed, fn.__name__
    nb = json.load(open(os.path.join(HERE, "experiment_harness.ipynb"), encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    m = re.search(r"OVERSAMPLE, REFINEMENTS = (\d+), (\d+)", src)
    assert (int(m.group(1)), int(m.group(2))) == (listed["s"], listed["h"])


def test_check_catches_a_name_mismatch():
    """The comparison fails on exactly the kind of error it exists for."""
    code = ast.unparse(function_in_text("matmul_launcher"))
    broken = parse(code.replace("cp.asarray(A,", "cp.asarray(A_I,")).body[0]
    import matmul_launcher as mod
    ok, _ = same_function(broken, function_in_repo(mod, "matmul_launcher"))
    assert not ok


# ------------------------------------------------------------ CUDA listings

def _cuda_tokens(code):
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"#[A-Z]\b", "", code)
    code = code[code.index('extern "C"'):] if 'extern "C"' in code else code
    return re.findall(r"[A-Za-z_]\w*|\d+\.\d*f?|\d+|\S", code)


def test_listing_2_4_matches_kernel_file():
    kernel = open(os.path.join(HERE, "matmulkernel.cu"), encoding="utf-8").read()
    assert _cuda_tokens(numbered_listings()["2.4"][1]) == _cuda_tokens(kernel)


def test_listing_2_5_inlines_the_same_kernel():
    code = numbered_listings()["2.5"][1]
    src = re.search(r'r"""(.*?)"""', code, re.S).group(1)
    assert _cuda_tokens(src) == _cuda_tokens(numbered_listings()["2.4"][1])


def test_listing_2_3_braces_balance():
    code = re.sub(r"//[^\n]*", "", numbered_listings()["2.3"][1])
    assert code.count("{") == code.count("}")
