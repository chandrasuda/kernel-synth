"""Convert ``ModuleCandidate`` records into self-contained RL env folders.

Each env is a flat directory the user (or an RL agent) can run directly:

    envs/<owner__repo>__<ClassName>/
        README.md       # task description, reward, run instructions
        env.json        # machine-readable metadata
        reference.py    # original class + imports lifted from its source file
        inputs.py       # auto-inferred __init__ / forward input generators
        solution.py     # scaffold the kernel-engineer (or RL agent) edits
        harness.py      # eval loop: correctness * speedup -> reward

The shape inference is intentionally simple. We parse the class ``__init__``
and ``forward`` signatures and hand back sensible defaults for the canonical
arg names (``dim`` / ``d_model`` / ``heads`` / etc.). Anything we cannot
infer is left as a ``# TODO`` for the user to fill in.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import ModuleCandidate, RepoRecord


# ---------------------------------------------------------------------------
# Public API


def env_name(record: RepoRecord, cand: ModuleCandidate) -> str:
    owner_repo = record.name.replace("/", "__")
    return _safe(f"{owner_repo}__{cand.class_name}")


def build_env(
    record: RepoRecord,
    cand: ModuleCandidate,
    *,
    envs_root: Path,
) -> Path:
    """Materialize one RL env folder. Returns the env directory path."""
    env_dir = envs_root / env_name(record, cand)
    env_dir.mkdir(parents=True, exist_ok=True)

    imports = _extract_imports(
        Path(record.local_path) / cand.file_path,
        class_source=cand.source_code,
    )
    init_sig = _parse_signature(cand.source_code, "__init__")
    forward_sig = _parse_signature(cand.source_code, "forward")

    _write_reference(env_dir / "reference.py", imports=imports, cand=cand)
    _write_inputs(env_dir / "inputs.py", init_sig=init_sig, forward_sig=forward_sig)
    _write_solution(env_dir / "solution.py", cand=cand)
    _write_harness(env_dir / "harness.py", cand=cand)
    _write_env_json(env_dir / "env.json", record=record, cand=cand)
    _write_readme(
        env_dir / "README.md",
        record=record,
        cand=cand,
        init_sig=init_sig,
        forward_sig=forward_sig,
    )
    return env_dir


def build_all(
    records: Iterable[RepoRecord],
    *,
    envs_root: Path,
) -> list[Path]:
    """Build envs for every candidate in every record."""
    envs_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for record in records:
        for cand in record.candidates:
            try:
                paths.append(build_env(record, cand, envs_root=envs_root))
            except Exception as e:  # noqa: BLE001
                # Skip but record the failure beside the manifest.
                fail = envs_root / env_name(record, cand)
                fail.mkdir(parents=True, exist_ok=True)
                (fail / "_BUILD_FAILED.txt").write_text(
                    f"build_env failed: {e!r}\n", encoding="utf-8"
                )
    write_index(records, envs_root)
    return paths


def write_index(records: Iterable[RepoRecord], envs_root: Path) -> Path:
    """Write a top-level README + envs.json so the folder is GitHub-pushable."""
    rows: list[str] = []
    json_rows: list[dict] = []
    records_list = list(records)
    total = sum(len(r.candidates) for r in records_list)
    for record in records_list:
        for cand in record.candidates:
            slug = env_name(record, cand)
            tags = ", ".join(cand.tags) if cand.tags else ""
            rows.append(
                f"| [`{cand.class_name}`](./{slug}) | `{record.name}` | "
                f"{cand.novelty_score:.2f} | {tags} |"
            )
            json_rows.append(
                {
                    "name": slug,
                    "class_name": cand.class_name,
                    "repo": record.name,
                    "repo_url": record.url,
                    "source_file": cand.file_path,
                    "novelty_score": cand.novelty_score,
                    "tags": cand.tags,
                }
            )

    readme = textwrap.dedent(
        f"""\
        # kernel-synth · RL environments

        Auto-generated from extracted PyTorch modules. Each subfolder is a
        **self-contained kernel-engineering task**: write an optimized
        implementation of the reference module and beat it on latency while
        staying numerically equivalent.

        - **Envs:** {total}
        - **Source repos:** {len(records_list)}

        ## Layout per env

        ```
        <env_name>/
          README.md       # task spec
          env.json        # metadata
          reference.py    # original module (with imports from its source file)
          inputs.py       # input generators (EDIT for your shapes/dtypes)
          solution.py     # your starting point — edit this
          harness.py      # python harness.py --json  ->  reward JSON
        ```

        ## Reward

        ```
        reward = is_correct * clamp(speedup, 0, 10) / 10
        ```

        ## Running

        ```bash
        cd <env_name>
        pip install -r ../../requirements-env.txt   # torch + the source repo
        python harness.py --json
        ```

        ## Envs

        | class | repo | novelty | tags |
        |-------|------|--------:|------|
        """
    )
    readme += "\n".join(rows) + "\n"
    (envs_root / "README.md").write_text(readme, encoding="utf-8")
    (envs_root / "envs.json").write_text(
        json.dumps({"envs": json_rows}, indent=2), encoding="utf-8"
    )

    requirements = textwrap.dedent(
        """\
        # Generic deps for running any kernel-synth env harness.
        # Source-repo-specific deps (e.g. mamba_ssm, causal_conv1d) need to be
        # installed manually for envs that import them.
        torch>=2.1
        einops>=0.7
        numpy>=1.24
        """
    )
    (envs_root / "requirements-env.txt").write_text(requirements, encoding="utf-8")

    (envs_root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    return envs_root / "README.md"


# ---------------------------------------------------------------------------
# Helpers — source inspection


@dataclass
class SigInfo:
    args: list[str]                      # positional arg names (sans 'self')
    defaults: dict[str, str]             # arg -> repr(default) for those that have one
    has_args: bool = False               # *args present
    has_kwargs: bool = False             # **kwargs present


def _parse_signature(source: str, method_name: str) -> SigInfo | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for body in node.body:
            if (
                isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef))
                and body.name == method_name
            ):
                return _build_sig(body.args)
    return None


def _build_sig(args: ast.arguments) -> SigInfo:
    arg_names = [a.arg for a in args.args if a.arg != "self"]
    arg_names += [a.arg for a in (args.kwonlyargs or [])]
    defaults: dict[str, str] = {}

    pos_defaults = list(args.defaults or [])
    pos_args = [a.arg for a in args.args if a.arg != "self"]
    for name, default in zip(pos_args[-len(pos_defaults):], pos_defaults):
        defaults[name] = _unparse(default)
    for kw, default in zip(args.kwonlyargs or [], args.kw_defaults or []):
        if default is not None:
            defaults[kw.arg] = _unparse(default)

    return SigInfo(
        args=arg_names,
        defaults=defaults,
        has_args=args.vararg is not None,
        has_kwargs=args.kwarg is not None,
    )


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "None"


def _extract_imports(source_file: Path, class_source: str | None = None) -> str:
    """Return imports + module-level helper defs (functions / constants /
    smaller helper classes) from ``source_file`` whose bound names are
    actually referenced by ``class_source``.

    This avoids dragging in unused deps like ``loguru``, while still pulling
    in repo-local helpers like ``exists()`` / ``default()`` that the class
    relies on.
    """
    if not source_file.is_file():
        return _DEFAULT_IMPORT_HEADER

    try:
        text = source_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return _DEFAULT_IMPORT_HEADER

    used_names = _names_referenced(class_source) if class_source else None
    if used_names is None:
        used_names = set()
    # Add the class's own name so we don't accidentally re-pull itself.
    self_class_name = _first_class_name(class_source) if class_source else None

    lines = text.splitlines()
    import_lines: list[str] = []
    helper_lines: list[str] = []

    # Walk the file body in two passes so helpers we *transitively* need
    # (e.g. helpers used by other helpers) also come along.
    needed = set(used_names)
    body = list(tree.body)
    # Iterate to a fixed point on helper expansion.
    for _ in range(3):
        before = len(needed)
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in needed:
                    needed.update(_names_referenced_node(node))
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in needed:
                        needed.update(_names_referenced_node(node.value))
            elif isinstance(node, ast.ClassDef):
                if node.name in needed and node.name != self_class_name:
                    needed.update(_names_referenced_node(node))
        if len(needed) == before:
            break

    seen_ranges: set[tuple[int, int]] = set()
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = _imported_names(node)
            is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
            if is_future or (bound & needed):
                end = getattr(node, "end_lineno", node.lineno)
                snippet = "\n".join(lines[node.lineno - 1 : end])
                import_lines.append(snippet)
            continue

        # Module-level helpers the class needs.
        keep = False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            keep = node.name in needed
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in needed:
                    keep = True
                    break
        elif isinstance(node, ast.ClassDef):
            keep = node.name in needed and node.name != self_class_name

        if keep:
            end = getattr(node, "end_lineno", node.lineno)
            rng = (node.lineno, end)
            if rng in seen_ranges:
                continue
            seen_ranges.add(rng)
            helper_lines.append("\n".join(lines[node.lineno - 1 : end]))

    out = _DEFAULT_IMPORT_HEADER
    if import_lines:
        out += "\n# --- imports lifted from the source file (filtered) ---\n"
        out += "\n".join(import_lines) + "\n"
    if helper_lines:
        out += (
            "\n# --- helper definitions lifted from the source file"
            " (used by the class) ---\n"
        )
        out += "\n\n".join(helper_lines) + "\n"
    return out


def _names_referenced_node(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            inner: ast.AST = sub
            while isinstance(inner, ast.Attribute):
                inner = inner.value
            if isinstance(inner, ast.Name):
                out.add(inner.id)
    return out


def _first_class_name(source: str | None) -> str | None:
    if not source:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node.name
    return None


def _imported_names(node: ast.AST) -> set[str]:
    """Return the local names a single Import/ImportFrom node binds."""
    out: set[str] = set()
    if isinstance(node, ast.Import):
        for n in node.names:
            out.add((n.asname or n.name).split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for n in node.names:
            out.add(n.asname or n.name)
    return out


def _names_referenced(source: str) -> set[str]:
    """Return the set of bare names referenced anywhere in ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Walk to the leftmost Name.
            inner: ast.AST = node
            while isinstance(inner, ast.Attribute):
                inner = inner.value
            if isinstance(inner, ast.Name):
                out.add(inner.id)
    return out


_DEFAULT_IMPORT_HEADER = textwrap.dedent(
    """\
    # Auto-generated by kernel-synth env_builder.
    # If imports fail, install the source repo or stub the missing names.
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    """
)


# ---------------------------------------------------------------------------
# Shape inference heuristics


_INIT_NUMERIC_DEFAULTS = {
    "dim": 64, "d_model": 64, "hidden_size": 64, "embed_dim": 64, "n_embd": 64,
    "input_dim": 64, "output_dim": 64, "in_features": 64, "out_features": 64,
    "heads": 4, "n_heads": 4, "num_heads": 4, "n_head": 4,
    "head_dim": 16, "dim_head": 16,
    "depth": 2, "n_layers": 2, "num_layers": 2, "layers": 2,
    "max_pos": 64, "max_len": 64, "max_seq_len": 64, "max_position": 64,
    "vocab_size": 1024, "num_classes": 10, "num_tokens": 1024,
    "image_size": 32, "patch_size": 4, "channels": 3, "in_channels": 3, "out_channels": 16,
    "d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": "'auto'", "headdim": 16,
    "kernel_size": 3, "stride": 1, "padding": 0,
}

_INIT_BOOL_DEFAULTS = {
    "bias": False, "use_fast_path": False, "causal": False, "soft_onehot": False,
    "talking_heads": False, "qkv_bias": False, "use_cache": False,
}

# (arg_name_or_pattern -> generator expression as Python source).
_FORWARD_TENSOR_GENS: list[tuple[str, str]] = [
    (r"^(x|input|hidden_states|inputs_embeds|features|h)$",
     "torch.randn(BATCH, SEQ_LEN, HIDDEN)"),
    (r"^(query|q)$", "torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)"),
    (r"^(key|k)$",   "torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)"),
    (r"^(value|v)$", "torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)"),
    (r"^attn_logits$", "torch.randn(BATCH, HEADS, SEQ_LEN, SEQ_LEN)"),
    (r"^(mask|attention_mask|pad_mask)$",
     "torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)"),
    (r"^(input_ids|tokens|ids)$",
     "torch.randint(0, 1024, (BATCH, SEQ_LEN))"),
    (r"^(audio|waveform|wav)$",
     "torch.randn(BATCH, 16000)"),
    (r"^(image|img|pixel_values|images)$",
     "torch.randn(BATCH, 3, 32, 32)"),
    (r"^(position_ids|positions)$",
     "torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)"),
    (r"^t$|^time$|^timestep$|^step$",
     "torch.zeros(BATCH, dtype=torch.long)"),
]


def _guess_init_kwargs(sig: SigInfo | None) -> dict[str, str]:
    """Return a dict of ``{kwarg_name: repr_value}`` for the module ``__init__``."""
    if sig is None:
        return {}
    out: dict[str, str] = {}
    for name in sig.args:
        if name in sig.defaults:
            continue  # let the default fire — we don't need to pass it
        if name in _INIT_NUMERIC_DEFAULTS:
            out[name] = repr(_INIT_NUMERIC_DEFAULTS[name])
        elif name in _INIT_BOOL_DEFAULTS:
            out[name] = repr(_INIT_BOOL_DEFAULTS[name])
        else:
            out[name] = f"None  # TODO: pick a value for {name!r}"
    return out


def _guess_forward_inputs(sig: SigInfo | None) -> list[tuple[str, str]]:
    """Return [(name, generator_source)] for the forward call."""
    if sig is None:
        return [("x", "torch.randn(BATCH, SEQ_LEN, HIDDEN)")]
    out: list[tuple[str, str]] = []
    for name in sig.args:
        if name in sig.defaults:
            continue  # skip kwargs in the call by default
        gen = _generator_for(name)
        out.append((name, gen))
    if not out:
        out.append(("x", "torch.randn(BATCH, SEQ_LEN, HIDDEN)"))
    return out


def _generator_for(name: str) -> str:
    for pattern, gen in _FORWARD_TENSOR_GENS:
        if re.fullmatch(pattern, name):
            return gen
    return f"torch.randn(BATCH, SEQ_LEN, HIDDEN)  # TODO: shape for {name!r}"


# ---------------------------------------------------------------------------
# File writers


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


_FUTURE_RE = re.compile(r"^from\s+__future__\s+import[^\n]*\n?", re.MULTILINE)


def _write_reference(path: Path, *, imports: str, cand: ModuleCandidate) -> None:
    body = cand.source_code or ""
    # __future__ imports MUST be the first statements in the file. Hoist any
    # that appear in the lifted import block (or in the class body) above the
    # docstring.
    futures: list[str] = []
    for src in (imports, body):
        for m in _FUTURE_RE.finditer(src):
            line = m.group(0).strip()
            if line not in futures:
                futures.append(line)
    imports_clean = _FUTURE_RE.sub("", imports)
    body_clean = _FUTURE_RE.sub("", body)

    future_block = ("\n".join(futures) + "\n\n") if futures else ""
    text = (
        f"{future_block}"
        f'"""Reference implementation of {cand.class_name}.\n'
        f"Source: {cand.file_path} (lines {cand.start_line}-{cand.end_line}).\n"
        f'Do not modify — your optimized version belongs in solution.py.\n"""\n'
        f"{imports_clean}\n"
        f"{body_clean}\n"
    )
    path.write_text(text, encoding="utf-8")


def _write_inputs(
    path: Path,
    *,
    init_sig: SigInfo | None,
    forward_sig: SigInfo | None,
) -> None:
    init_kwargs = _guess_init_kwargs(init_sig)
    forward_inputs = _guess_forward_inputs(forward_sig)

    kw_lines = ",\n        ".join(f"{k}={v}" for k, v in init_kwargs.items())
    fwd_gens = "\n    ".join(f"{name} = {gen}" for name, gen in forward_inputs)
    fwd_returns = ", ".join(name for name, _ in forward_inputs)

    text = textwrap.dedent(f'''\
        """Input generators for the eval harness.

        EDIT these to match the module's expected shapes / dtypes.
        Auto-inferred defaults are best-effort.
        """
        import torch

        # Shared problem-size knobs. Tune these for your benchmark.
        BATCH = 2
        SEQ_LEN = 64
        HIDDEN = 64
        HEADS = 4
        HEAD_DIM = HIDDEN // HEADS
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DTYPE = torch.float32


        def build_module_kwargs() -> dict:
            """Constructor kwargs for the reference / solution module."""
            return dict(
        ''') + (f"        {kw_lines},\n" if kw_lines else "") + textwrap.dedent('''\
            )


        def build_forward_inputs() -> tuple[tuple, dict]:
            """Return (positional args, keyword args) for the forward call."""
        ''') + textwrap.indent(f"    {fwd_gens}\n    return ({fwd_returns},), {{}}\n", "")
    path.write_text(text, encoding="utf-8")


def _write_solution(path: Path, *, cand: ModuleCandidate) -> None:
    text = textwrap.dedent(f'''\
        """Your optimized implementation of {cand.class_name}.

        Replace the body of ``build`` with anything that:

        * accepts the same constructor kwargs as the reference module
        * is callable like ``module(*args, **kwargs)``
        * returns the same shape & dtype, numerically equivalent

        You may use: pure PyTorch, ``torch.compile``, Triton, custom CUDA, etc.
        Don't touch ``reference.py`` or ``inputs.py``.
        """
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))

        import reference  # noqa: E402


        def build(**kwargs):
            """Return a module instance to be benchmarked.

            The default returns the reference verbatim — your starting baseline,
            with speedup = 1.0. Beat it.
            """
            return reference.{cand.class_name}(**kwargs)
        ''')
    path.write_text(text, encoding="utf-8")


def _write_harness(path: Path, *, cand: ModuleCandidate) -> None:
    text = textwrap.dedent(f'''\
        """Eval harness for the {cand.class_name} kernel-engineering env.

        Usage:
            python harness.py            # human-readable result
            python harness.py --json     # machine-readable JSON
            python harness.py --runs 50  # average over more timing samples
        """
        from __future__ import annotations

        import argparse
        import json
        import sys
        import time
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))

        import torch  # noqa: E402

        import reference  # noqa: E402
        import solution as solution_mod  # noqa: E402
        from inputs import (  # noqa: E402
            BATCH, SEQ_LEN, HIDDEN, HEADS, HEAD_DIM, DEVICE, DTYPE,
            build_module_kwargs, build_forward_inputs,
        )


        def _to(obj, device, dtype):
            if isinstance(obj, torch.Tensor):
                if obj.dtype.is_floating_point:
                    return obj.to(device=device, dtype=dtype)
                return obj.to(device=device)
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to(x, device, dtype) for x in obj)
            if isinstance(obj, dict):
                return {{k: _to(v, device, dtype) for k, v in obj.items()}}
            return obj


        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()


        def _time(module, args, kwargs, runs: int) -> tuple[object, float]:
            with torch.no_grad():
                for _ in range(max(2, runs // 5)):
                    out = module(*args, **kwargs)
            _sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(runs):
                    out = module(*args, **kwargs)
            _sync()
            return out, (time.perf_counter() - t0) / runs


        def _allclose(a, b, rtol=1e-3, atol=1e-4):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.shape != b.shape or a.dtype != b.dtype:
                    return False, float("inf")
                diff = (a - b).abs().max().item()
                return diff < atol + rtol * b.abs().max().item(), diff
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
                worst = 0.0
                ok = True
                for x, y in zip(a, b):
                    okx, dx = _allclose(x, y, rtol, atol)
                    ok = ok and okx
                    worst = max(worst, dx)
                return ok, worst
            return a == b, 0.0


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(description=__doc__)
            parser.add_argument("--json", action="store_true")
            parser.add_argument("--runs", type=int, default=20)
            args = parser.parse_args(argv)

            torch.manual_seed(0)

            try:
                kwargs = build_module_kwargs()
                fwd_args, fwd_kwargs = build_forward_inputs()
            except Exception as e:  # noqa: BLE001
                msg = {{"error": "input_build_failed", "detail": repr(e)}}
                print(json.dumps(msg, indent=2))
                return 2

            try:
                ref_mod = reference.{cand.class_name}(**kwargs).to(DEVICE, DTYPE)
                ref_mod.eval()
            except Exception as e:  # noqa: BLE001
                msg = {{"error": "reference_init_failed", "detail": repr(e)}}
                print(json.dumps(msg, indent=2))
                return 3

            try:
                cand_mod = solution_mod.build(**kwargs).to(DEVICE, DTYPE)
                cand_mod.eval()
            except Exception as e:  # noqa: BLE001
                msg = {{"error": "solution_init_failed", "detail": repr(e)}}
                print(json.dumps(msg, indent=2))
                return 4

            fwd_args = _to(fwd_args, DEVICE, DTYPE)
            fwd_kwargs = _to(fwd_kwargs, DEVICE, DTYPE)

            try:
                ref_out, ref_t = _time(ref_mod, fwd_args, fwd_kwargs, args.runs)
                cand_out, cand_t = _time(cand_mod, fwd_args, fwd_kwargs, args.runs)
            except Exception as e:  # noqa: BLE001
                msg = {{"error": "forward_failed", "detail": repr(e)}}
                print(json.dumps(msg, indent=2))
                return 5

            correct, diff = _allclose(ref_out, cand_out)
            speedup = ref_t / cand_t if cand_t > 0 else 0.0
            reward = float(correct) * max(0.0, min(speedup, 10.0)) / 10.0

            result = {{
                "module": "{cand.class_name}",
                "correct": bool(correct),
                "max_diff": float(diff),
                "ref_ms": ref_t * 1000,
                "cand_ms": cand_t * 1000,
                "speedup": speedup,
                "reward": reward,
                "device": DEVICE,
                "dtype": str(DTYPE),
                "shapes": {{
                    "BATCH": BATCH, "SEQ_LEN": SEQ_LEN, "HIDDEN": HIDDEN,
                    "HEADS": HEADS, "HEAD_DIM": HEAD_DIM,
                }},
            }}
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for k, v in result.items():
                    print(f"{{k:>10s}}  {{v}}")
            return 0 if correct else 1


        if __name__ == "__main__":
            raise SystemExit(main())
        ''')
    path.write_text(text, encoding="utf-8")


def _write_env_json(
    path: Path, *, record: RepoRecord, cand: ModuleCandidate
) -> None:
    payload = {
        "name": env_name(record, cand),
        "class_name": cand.class_name,
        "source": {
            "repo": record.name,
            "url": record.url,
            "commit_sha": record.commit_sha,
            "file_path": cand.file_path,
            "start_line": cand.start_line,
            "end_line": cand.end_line,
        },
        "novelty_score": cand.novelty_score,
        "tags": cand.tags,
        "reason": cand.reason,
        "selection_mode": record.selection_mode,
        "reward": {
            "formula": "is_correct * clamp(speedup, 0, 10) / 10",
            "correctness_rtol": 1e-3,
            "correctness_atol": 1e-4,
        },
        "version": "0.1.0",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_readme(
    path: Path,
    *,
    record: RepoRecord,
    cand: ModuleCandidate,
    init_sig: SigInfo | None,
    forward_sig: SigInfo | None,
) -> None:
    init_sig_str = _format_sig(init_sig)
    fwd_sig_str = _format_sig(forward_sig)
    tags = ", ".join(f"`{t}`" for t in cand.tags) if cand.tags else "—"
    text = textwrap.dedent(
        f"""\
        # `{cand.class_name}`  ·  RL kernel-engineering env

        > {cand.reason}

        **Source:** [`{record.name}`]({record.url}) · `{cand.file_path}`
        (lines {cand.start_line}–{cand.end_line})
        **Novelty:** {cand.novelty_score:.2f}
        **Tags:** {tags}
        **Selection mode:** `{record.selection_mode}`

        ## Goal

        Write an implementation of `{cand.class_name}` in `solution.py` that is
        numerically equivalent to `reference.py` but **faster**. The harness
        rewards correctness × clipped speedup.

        ## Signatures (inferred)

        ```python
        {cand.class_name}.__init__({init_sig_str})
        {cand.class_name}.forward({fwd_sig_str})
        ```

        ## Files

        - `reference.py` — frozen original implementation (do not edit)
        - `inputs.py` — input generators (**edit if defaults don't fit**)
        - `solution.py` — your implementation
        - `harness.py` — eval loop
        - `env.json` — machine-readable metadata

        ## Run

        ```bash
        # one-off
        python harness.py

        # machine-readable
        python harness.py --json
        ```

        ## Reward

        ```
        reward = float(correct) * clamp(speedup, 0, 10) / 10
        ```

        Where:
        - `correct = allclose(ref_out, cand_out, rtol=1e-3, atol=1e-4)`
        - `speedup = ref_latency / cand_latency`

        ## Notes

        - The reference imports were lifted verbatim from the source file. If
          your environment is missing a source-repo dep (e.g. `mamba_ssm`,
          `causal_conv1d`), `pip install` it or stub the missing names.
        - Default shapes are deliberately small so the harness runs on CPU.
          Increase `BATCH` / `SEQ_LEN` / `HIDDEN` in `inputs.py` for a
          meaningful benchmark on GPU.
        """
    )
    path.write_text(text, encoding="utf-8")


def _format_sig(sig: SigInfo | None) -> str:
    if sig is None:
        return "...args inferred from source..."
    parts: list[str] = []
    for name in sig.args:
        if name in sig.defaults:
            parts.append(f"{name}={sig.defaults[name]}")
        else:
            parts.append(name)
    if sig.has_args:
        parts.append("*args")
    if sig.has_kwargs:
        parts.append("**kwargs")
    return ", ".join(parts) if parts else ""
