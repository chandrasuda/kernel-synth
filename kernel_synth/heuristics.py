"""Deterministic AST-based fallback for picking unique nn.Module classes.

Used when no LLM API key is available, so the pipeline always produces a
usable RL-environment buffer end-to-end. The signal is intentionally
*opinionated*: we down-weight modules that look like vanilla Attention /
FeedForward / Linear stacks and up-weight modules that touch unusual ops
(scan, gating, custom einsum, triton/cuda imports, learned positional
codes, etc.).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .code_buffer import CodeBuffer, FileEntry, slice_source
from .models import ModuleCandidate

# Names that — by themselves — strongly suggest "already well-served by
# existing kernels". These get a uniqueness penalty.
_GENERIC_NAME_PARTS = {
    "attention",
    "selfattention",
    "crossattention",
    "multihead",
    "mlp",
    "feedforward",
    "ffn",
    "linear",
    "conv",
    "layernorm",
    "batchnorm",
    "embedding",
    "dropout",
    "block",
    "encoder",
    "decoder",
    "transformer",
}

# Calls that signal "interesting kernel territory" if they show up in forward().
_INTERESTING_OPS = {
    "einsum",
    "scan",
    "associative_scan",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "gather",
    "index_select",
    "softmax",  # context dependent; weighted below
    "logsumexp",
    "complex",
    "fft",
    "irfft",
    "rfft",
    "cumprod",
    "cumsum",
    "topk",
    "sort",
    "unique",
    "roll",
    "narrow",
    "as_strided",
    "diag",
    "tril",
    "triu",
    "kron",
    "matrix_exp",
    "lu_solve",
    "cholesky",
    "qr",
    "svd",
    "lstsq",
    "linalg",
    "selective_scan",
    "rms_norm",
    "silu",
    "gelu",
    "swish",
    "rotary",
    "rope",
    "lerp",
    "addcmul",
    "addmm",
    "baddbmm",
    "bmm",
    # MoE-shaped ops. Routing / dispatch / expert math is exactly the kind
    # of irregular workload where a custom kernel pays for itself, so we
    # want the heuristic to surface modules that touch any of these names.
    "top_k",
    "gating",
    "gate",
    "expert",
    "experts",
    "router",
    "routing",
    "dispatch",
    "combine",
    "load_balance",
}

# Substrings in (lowercased) class names that suggest a non-standard
# mechanism. Order doesn't matter; we OR-merge into one regex.
_CREATIVE_NAME_TOKENS = (
    "mamba", "ssm", "selective",
    # Subquadratic / structured-attention families.
    "hyena", "monarch", "performer", "linformer", "longformer", "linear_attention",
    "linearattention", "linear_attn", "linearattn", "retentive", "retnet",
    "rwkv", "bytenet", "lambda_layer", "lambdalayer", "synthesizer",
    "fnet", "afno", "gss",
    # MoE / routing.
    "moe", "gated", "gating", "router", "expert", "dispatch",
    # Custom kernels / numerics.
    "kernel", "fourier", "wavelet", "rotary", "alibi",
    "deformable", "sparse", "hash", "hyperbolic", "spline",
    # Bio-inspired / continuous-time.
    "liquid", "spiking",
)
_CREATIVE_NAME_RE = re.compile("|".join(_CREATIVE_NAME_TOKENS))


_INTERESTING_IMPORTS = {
    "triton",
    "einops",
    "flash_attn",
    "xformers",
    "deepspeed",
    "apex",
    "torch.utils.checkpoint",
    "torch.fx",
    "torch.compile",
    "selective_scan_cuda",
    "causal_conv1d",
    "mamba_ssm",
}


def select_candidates(
    buffer: CodeBuffer,
    *,
    repo_root: Path,
    top_k: int = 8,
) -> list[ModuleCandidate]:
    """Return up to ``top_k`` candidate :class:`ModuleCandidate` objects."""
    scored: list[tuple[float, ModuleCandidate]] = []

    for f in buffer.files_with_nn_modules():
        try:
            source = f.read()
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        file_imports = _imports(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not _has_nn_module_base(node):
                continue
            end_line = getattr(node, "end_lineno", node.lineno)
            class_src = slice_source(f, node.lineno, end_line)
            score, tags, reason = _score_class(node, class_src, file_imports)
            if score <= 0:
                continue
            cand = ModuleCandidate(
                file_path=f.path,
                class_name=node.name,
                start_line=node.lineno,
                end_line=end_line,
                reason=reason,
                novelty_score=min(score, 1.0),
                tags=tags,
                source_code=class_src,
            )
            scored.append((score, cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Light de-duplication: per-class name across files (lucidrains-style repos
    # tend to repeat names).
    seen: set[tuple[str, str]] = set()
    out: list[ModuleCandidate] = []
    for _, cand in scored:
        key = (cand.class_name, cand.file_path.split("/")[-1])
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= top_k:
            break
    return out


def _has_nn_module_base(node: ast.ClassDef) -> bool:
    for b in node.bases:
        s = _base_str(b)
        if s.endswith("nn.Module") or s == "Module" or s == "torch.nn.Module":
            return True
    return False


def _base_str(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_base_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _base_str(node.func)
    return ""


def _imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.add(mod)
            for n in node.names:
                out.add(f"{mod}.{n.name}" if mod else n.name)
    return out


def _score_class(
    node: ast.ClassDef,
    source: str,
    file_imports: set[str],
) -> tuple[float, list[str], str]:
    name_lower = node.name.lower()
    loc = source.count("\n") + 1

    score = 0.0
    tags: list[str] = []
    notes: list[str] = []

    # --- generic-name penalty ---------------------------------------------
    if any(part in name_lower for part in _GENERIC_NAME_PARTS):
        # The penalty is softer if the body is meaty.
        score -= 0.25
        tags.append("looks-generic")

    # --- size sweet spot ---------------------------------------------------
    # 30..400 LOC tends to be a self-contained custom block.
    if 30 <= loc <= 400:
        score += 0.25
        tags.append("self-contained")
    elif loc > 400:
        score += 0.1
    else:
        score -= 0.1

    # --- interesting ops in forward() -------------------------------------
    forward = _find_forward(node)
    if forward is not None:
        op_hits = _interesting_op_hits(forward)
        if op_hits:
            score += min(0.5, 0.08 * len(op_hits))
            tags.extend(sorted(op_hits))
            notes.append(f"forward uses {', '.join(sorted(op_hits))}")
        # Lots of arithmetic in forward = custom math.
        arith_ops = _count_arith(forward)
        if arith_ops >= 8:
            score += 0.15
            tags.append("math-heavy")
            notes.append(f"~{arith_ops} arithmetic ops in forward")

    # --- interesting imports at file level --------------------------------
    interesting_imp_hits = sorted(
        imp for imp in file_imports if _imp_is_interesting(imp)
    )
    if interesting_imp_hits:
        roots = sorted({imp.split(".")[0] for imp in interesting_imp_hits})
        score += min(0.4, 0.15 * len(roots))
        for root in roots:
            tags.append(f"imports:{root}")
        notes.append("file imports " + ", ".join(roots))

    # --- parameter / buffer creativity ------------------------------------
    n_params, n_buffers, weird_init = _param_buffer_stats(node)
    if n_buffers >= 1:
        score += 0.1
        tags.append("uses-buffers")
    if weird_init:
        score += 0.1
        tags.append("custom-init")
        notes.append("non-trivial parameter init")
    if n_params >= 4:
        score += 0.05

    # --- naming creativity bonus ------------------------------------------
    if _CREATIVE_NAME_RE.search(name_lower):
        score += 0.25
        tags.append("creative-name")
        notes.append(f"name '{node.name}' suggests a non-standard mechanism")

    if score <= 0:
        return 0.0, tags, ""

    reason = "; ".join(notes) or f"Custom nn.Module '{node.name}' with non-trivial body."
    return score, sorted(set(tags)), reason


def _find_forward(node: ast.ClassDef) -> ast.FunctionDef | None:
    for body_node in node.body:
        if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if body_node.name == "forward":
                return body_node  # type: ignore[return-value]
    return None


def _interesting_op_hits(forward: ast.AST) -> set[str]:
    hits: set[str] = set()
    for node in ast.walk(forward):
        if isinstance(node, ast.Call):
            name = _call_short_name(node.func)
            if name and name.lower() in _INTERESTING_OPS:
                hits.add(name.lower())
    return hits


def _count_arith(forward: ast.AST) -> int:
    n = 0
    for node in ast.walk(forward):
        if isinstance(node, ast.BinOp):
            n += 1
        elif isinstance(node, ast.AugAssign):
            n += 1
    return n


def _call_short_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _imp_is_interesting(imp: str) -> bool:
    if not imp:
        return False
    lowered = imp.lower()
    return any(lowered == n or lowered.startswith(n + ".") for n in _INTERESTING_IMPORTS)


def _param_buffer_stats(node: ast.ClassDef) -> tuple[int, int, bool]:
    n_params = 0
    n_buffers = 0
    weird_init = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = _call_short_name(sub.func)
            if name == "Parameter":
                n_params += 1
                for arg in sub.args:
                    if isinstance(arg, ast.Call):
                        inner = _call_short_name(arg.func)
                        if inner and inner not in {"zeros", "ones", "empty"}:
                            weird_init = True
            elif name == "register_buffer":
                n_buffers += 1
    return n_params, n_buffers, weird_init
