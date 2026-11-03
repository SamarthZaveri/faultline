"""Small AST helpers shared by detectors and the injector."""

from __future__ import annotations

import ast
from collections.abc import Iterator


def _all(tree: ast.AST | list[ast.AST]) -> list[ast.AST] | Iterator[ast.AST]:
    return tree if isinstance(tree, list) else ast.walk(tree)


def import_aliases(tree: ast.AST | list[ast.AST]) -> dict[str, str]:
    """Local name -> fully qualified name, e.g. {'pd': 'pandas', 'load': 'joblib.load'}."""
    out: dict[str, str] = {}
    for node in _all(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    head = a.name.split(".", 1)[0]
                    out[head] = head
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


def dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def resolve(name: str, aliases: dict[str, str]) -> str:
    head, sep, rest = name.partition(".")
    full = aliases.get(head, head)
    return f"{full}{sep}{rest}"


def call_name(call: ast.Call, aliases: dict[str, str]) -> str | None:
    """Fully qualified callee, e.g. `pk.loads(x)` -> 'pickle.loads'. None for computed callees."""
    d = dotted(call.func)
    return resolve(d, aliases) if d else None


def last_segment(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def arg(call: ast.Call, index: int, *names: str) -> ast.expr | None:
    if len(call.args) > index and not isinstance(call.args[index], ast.Starred):
        return call.args[index]
    for n in names:
        v = kwarg(call, n)
        if v is not None:
            return v
    return None


def is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def is_false(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def const_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def is_dynamic_str(node: ast.expr) -> bool:
    """A string built at runtime from non-constant parts (f-string, %, +, .format)."""
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return const_str(node.left) is not None or is_dynamic_str(node.left)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        parts = (node.left, node.right)
        has_str = any(const_str(p) is not None or isinstance(p, (ast.JoinedStr, ast.BinOp)) for p in parts)
        has_var = any(const_str(p) is None and not isinstance(p, ast.Constant) for p in parts)
        return has_str and has_var
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return bool(node.args or node.keywords)
    return False


def str_fragments(node: ast.expr) -> str:
    """Concatenate the constant text inside a (possibly dynamic) string expression."""
    return " ".join(
        n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    )


def names_in(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def target_names(target: ast.expr) -> set[str]:
    """Names bound (or mutated, for `df['x'] = ...`) by an assignment target."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(target_names(e) for e in target.elts)) if target.elts else set()
    if isinstance(target, ast.Starred):
        return target_names(target.value)
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        base: ast.expr = target
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        return {base.id} if isinstance(base, ast.Name) else set()
    return set()


def latest_assignments(tree: ast.AST | list[ast.AST]) -> dict[str, list[tuple[int, ast.expr]]]:
    """name -> [(line, value)] for simple `name = value` assignments, in source order."""
    out: dict[str, list[tuple[int, ast.expr]]] = {}
    for node in _all(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out.setdefault(node.targets[0].id, []).append((node.lineno, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            out.setdefault(node.target.id, []).append((node.lineno, node.value))
    for v in out.values():
        v.sort(key=lambda t: t[0])
    return out


def value_before(assigns: dict[str, list[tuple[int, ast.expr]]], name: str, line: int) -> ast.expr | None:
    best = None
    for ln, value in assigns.get(name, []):
        if ln <= line:
            best = value
    return best
