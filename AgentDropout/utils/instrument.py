"""Per-call and per-edge token instrumentation.

Writes newline-delimited JSON to $INSTRUMENT_DIR (default ./instrument_logs):

  calls.jsonl        one row per LLM call, counts reported by the server
  edges.jsonl        one row per contributing predecessor, rendered segment size
  prompt_build.jsonl one row per node execution, base vs full prompt size
  edges_error.jsonl  renderer failures, so silent gaps are visible

Call identity travels by contextvar. Node.async_execute sets the vars
immediately before asyncio.create_task, which snapshots the context, so
concurrently executing nodes each see their own values.
"""
import contextvars
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading

import tiktoken

CTX_QUESTION = contextvars.ContextVar("question_id", default=None)
CTX_ROUND = contextvars.ContextVar("round_idx", default=-1)
CTX_NODE_ID = contextvars.ContextVar("node_id", default=None)
CTX_NODE_NAME = contextvars.ContextVar("node_name", default=None)
CTX_ROLE = contextvars.ContextVar("role", default=None)
CTX_IS_DEC = contextvars.ContextVar("is_decision", default=False)
CTX_RUN_TAG = contextvars.ContextVar(
    "run_tag", default=os.path.basename(os.environ.get("INSTRUMENT_DIR", "").rstrip("/")) or "")

_lock = threading.Lock()
_handles = {}
_ENC = None
_manifest_done = False


def count_tokens(text):
    """Local token count, used only for the per-edge segment breakdown.

    The server's usage field stays authoritative for totals; this is for the
    relative split between senders, which the API does not report.
    """
    global _ENC
    if _ENC is None:
        model = os.environ.get("INSTRUMENT_MODEL", "gpt-4o-mini")
        try:
            _ENC = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENC = tiktoken.get_encoding("o200k_base")
    return len(_ENC.encode(text or ""))


def qid(raw_inputs):
    """Stable id for a question, so rows can be grouped without a counter."""
    task = raw_inputs.get("task", "") if isinstance(raw_inputs, dict) else str(raw_inputs)
    return hashlib.sha1(task.encode("utf-8")).hexdigest()[:12]


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=_repo_root(),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _write_manifest(directory):
    """Provenance for one run. Written once, on the first logged row.

    Without this a log directory is unreadable a month later: you cannot tell
    which topology, model, dataset or code version produced it.
    """
    try:
        from AgentDropout.utils.globals import Time
        run_time = Time.instance().value
    except Exception:
        run_time = None
    pkgs = {}
    for name in ("openai", "torch", "tiktoken", "transformers"):
        try:
            pkgs[name] = __import__(name).__version__
        except Exception:
            pkgs[name] = None
    manifest = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_tag": CTX_RUN_TAG.get(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "env": {k: os.environ.get(k) for k in ("INSTRUMENT_DIR", "INSTRUMENT_MODEL", "PYTHONPATH")},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "repo_time_stamp": run_time,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
    }
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def log(stream, **row):
    row.setdefault("question_id", CTX_QUESTION.get())
    row.setdefault("round", CTX_ROUND.get())
    row.setdefault("run_tag", CTX_RUN_TAG.get())
    directory = os.environ.get("INSTRUMENT_DIR", "instrument_logs")
    global _manifest_done
    with _lock:
        if not _manifest_done:
            _manifest_done = True
            os.makedirs(directory, exist_ok=True)
            try:
                _write_manifest(directory)
            except Exception as exc:
                print(f"[instrument] manifest failed: {exc!r}")
        if stream not in _handles:
            os.makedirs(directory, exist_ok=True)
            _handles[stream] = open(
                os.path.join(directory, f"{stream}.jsonl"), "a", encoding="utf-8"
            )
        _handles[stream].write(json.dumps(row, ensure_ascii=False) + "\n")
        _handles[stream].flush()
