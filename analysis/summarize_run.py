#!/usr/bin/env python3
"""Summarize one instrumented run, and write summary.json beside its logs.

    python analysis/summarize_run.py instrument_logs/smoke01
    python analysis/summarize_run.py --all          # one row per run

Correctness lives in the repo's own result JSON, which has no link back to the
logs. We join on a hash of the task text, which is how question_id is built,
so the join survives renamed or re-timestamped result files.
"""
import argparse, glob, hashlib, json, os, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows(d, name):
    p = os.path.join(d, f"{name}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def _correctness(qids):
    """Best-matching result file, as {question_id: solved}."""
    best = {}
    for path in glob.glob(os.path.join(ROOT, "result", "*", "*.json")):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        got = {}
        for item in data if isinstance(data, list) else []:
            task = (item.get("Question") or {}).get("task")
            if task is None or "Solved" not in item:
                continue
            got[hashlib.sha1(task.encode("utf-8")).hexdigest()[:12]] = float(item["Solved"])
        hit = {k: v for k, v in got.items() if k in qids}
        if len(hit) > len(best):
            best = hit
    return best


def summarize(d):
    calls, edges = _rows(d, "calls"), _rows(d, "edges")
    if not calls:
        raise SystemExit(f"no calls.jsonl in {d}")
    manifest = {}
    mp = os.path.join(d, "manifest.json")
    if os.path.exists(mp):
        manifest = json.load(open(mp, encoding="utf-8"))

    qids = {c["question_id"] for c in calls}
    tok = lambda rs: sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs)
    dec = [c for c in calls if c["is_decision"]]
    total = tok(calls)
    solved = _correctness(qids)

    by_sender = defaultdict(list)
    for e in edges:
        by_sender[e["sender_role"]].append(e["sender_output_tokens"])
    segs = [e["rendered_segment_tokens"] for e in edges] or [0]

    # a repeated call is one whose prompt is byte-identical to an earlier round's
    seen, repeats = {}, []
    for c in sorted(calls, key=lambda c: (c["question_id"], c["round"])):
        key = (c["question_id"], c.get("node_id"), c.get("user_prompt"))
        if key in seen:
            repeats.append(c)
        seen[key] = True

    s = {
        "run": os.path.basename(os.path.normpath(d)),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "argv": manifest.get("argv"),
        "questions": len(qids),
        "calls": len(calls),
        "tokens_total": total,
        "tokens_prompt": sum(c["prompt_tokens"] for c in calls),
        "tokens_completion": sum(c["completion_tokens"] for c in calls),
        "tokens_per_question": round(total / max(len(qids), 1), 1),
        "decision_share_tokens": round(tok(dec) / max(total, 1), 4),
        "accuracy": round(sum(solved.values()) / len(solved), 4) if solved else None,
        "accuracy_n": len(solved),
        "segment_min": min(segs), "segment_median": st.median(segs), "segment_max": max(segs),
        "segment_mean": round(st.mean(segs), 1),
        "segment_max_over_median": round(max(segs) / max(st.median(segs), 1), 2),
        "mean_output_by_sender_role": {k: round(st.mean(v), 1) for k, v in sorted(
            by_sender.items(), key=lambda kv: -st.mean(kv[1]))},
        "repeated_calls": len(repeats),
        "repeated_call_token_share": round(tok(repeats) / max(total, 1), 4),
    }
    json.dump(s, open(os.path.join(d, "summary.json"), "w", encoding="utf-8"), indent=2)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--all", action="store_true", help="summarize every run under instrument_logs/")
    a = ap.parse_args()
    dirs = (sorted(glob.glob(os.path.join(ROOT, "instrument_logs", "*")))
            if a.all else [a.run_dir])
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        try:
            out.append(summarize(d))
        except SystemExit as e:
            print(e)
    if a.all:
        hdr = ["run", "questions", "accuracy", "tokens_per_question",
               "decision_share_tokens", "segment_max_over_median", "repeated_call_token_share"]
        print(" | ".join(h[:22].ljust(22) for h in hdr))
        for s in out:
            print(" | ".join(str(s.get(h))[:22].ljust(22) for h in hdr))
    else:
        print(json.dumps(out[0], indent=2))


if __name__ == "__main__":
    main()
