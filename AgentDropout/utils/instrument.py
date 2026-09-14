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
import hashlib
import json
import os
import threading

import tiktoken

CTX_QUESTION = contextvars.ContextVar("question_id", default=None)
CTX_ROUND = contextvars.ContextVar("round_idx", default=-1)
CTX_NODE_ID = contextvars.ContextVar("node_id", default=None)
CTX_NODE_NAME = contextvars.ContextVar("node_name", default=None)
CTX_ROLE = contextvars.ContextVar("role", default=None)
CTX_IS_DEC = contextvars.ContextVar("is_decision", default=False)
CTX_RUN_TAG = contextvars.ContextVar("run_tag", default="")

_lock = threading.Lock()
_handles = {}
_ENC = None


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


def log(stream, **row):
    row.setdefault("question_id", CTX_QUESTION.get())
    row.setdefault("round", CTX_ROUND.get())
    row.setdefault("run_tag", CTX_RUN_TAG.get())
    directory = os.environ.get("INSTRUMENT_DIR", "instrument_logs")
    with _lock:
        if stream not in _handles:
            os.makedirs(directory, exist_ok=True)
            _handles[stream] = open(
                os.path.join(directory, f"{stream}.jsonl"), "a", encoding="utf-8"
            )
        _handles[stream].write(json.dumps(row, ensure_ascii=False) + "\n")
        _handles[stream].flush()
