#!/usr/bin/env python3
"""PreToolUse guard for Bash commands in the integrator project.

Encodes the rules that have actually cost time or data here, as opposed to
general good practice. Two severities:

  deny   -- a rule with a known failure behind it; the call is blocked and the
            reason explains what to do instead.
  advise -- a lesson that applies to what is about to run but cannot be checked
            mechanically; passes the command through with a note.

Rules live in RULES so adding one is a single entry. Anything not matched here
passes straight through, so the hook stays cheap and quiet.
"""

import json
import re
import shlex
import sys
from pathlib import Path

HEREDOC = re.compile(
    r"<<-?\s*[\"\']?(\w+)[\"\']?\n.*?^\1\s*$",
    re.DOTALL | re.MULTILINE,
)


def _strip_heredocs(cmd: str) -> str:
    """Drop heredoc bodies before matching.

    A heredoc body is data being written to a file, not a command being
    run. Without this the guard fires on any script that merely mentions a
    forbidden pattern -- including its own test fixtures, which is exactly
    what happened the first time it ran.
    """
    return HEREDOC.sub("<<HEREDOC>>", cmd)


CLUSTER = re.compile(r"\bssh\s+cannon\b|/n/(netscratch|lab_storage|holylabs|hekstra)")


# the session's working directory, so relative paths resolve the way the
# command will see them rather than the way the hook process happens to
CWD = Path(".")


def _cluster(cmd: str) -> bool:
    return bool(CLUSTER.search(cmd))


def find_on_cluster(cmd: str):
    """`find` walks a shared filesystem and hammers the metadata servers."""
    if _cluster(cmd) and re.search(r"(^|[;&|]\s*|\bssh\b.*?)\bfind\s+/", cmd):
        return ("deny", "`find` on the shared cluster filesystem is not allowed. "
                "Use a targeted `ls` of a known path, or ask where the file lives.")


def bare_python(cmd: str):
    """Bare `python` picks up the wrong environment for PROJECT code.

    The rule exists so this project's modules import from its venv, so it
    applies to running a file in the repo or setting PYTHONPATH at it. A
    stdin script (`python3 -`), a `-c` one-liner, or a file outside the repo
    is stdlib glue and is left alone -- blocking those made the guard fire on
    its own test harness.
    """
    stripped = re.sub(r"/\S*/python[\d.]*", "", cmd)
    for m in re.finditer(r"(?:^|[;&|]\s*|\$\(\s*)python3?\s+(\S*)", stripped):
        arg = m.group(1)
        if arg.startswith("-"):          # python3 - , python3 -c, python3 -m
            if not arg.startswith("-m"):
                continue
        if re.search(r"uv\s+run[^\n;]*python", cmd):
            continue
        target = arg
        if target.startswith("-m") or target == "":
            return ("deny", "`python -m` on project code needs the project env: "
                    "use `uv run python -m ...`.")
        if not target.startswith("/") and not target.startswith("~"):
            return ("deny", "Run project Python through `uv run python`, not bare "
                    "`python`/`python3`.")
    if re.search(r"PYTHONPATH=\S*src\b", cmd) and not re.search(r"uv\s+run", cmd) \
            and not re.search(r"/\S*/python[\d.]*", cmd):
        return ("deny", "PYTHONPATH points at the project's src but the interpreter "
                "is not the project env: use `uv run python`.")


def git_add_broad(cmd: str):
    """`git add -A` and directory paths sweep in untracked files. Twice now."""
    for m in re.finditer(r"git\s+add\s+([^\n;&|]*)", cmd):
        args = m.group(1).strip()
        try:
            parts = shlex.split(args)
        except ValueError:
            parts = args.split()
        for p in parts:
            if p in ("-A", "--all", ".", "-a"):
                return ("deny", f"`git add {p}` sweeps in untracked files. "
                        "Stage tracked changes with `git add -u`, or name files "
                        "explicitly.")
            if not p.startswith("-") and (CWD / p).is_dir():
                return ("deny", f"`git add {p}` is a directory and will include "
                        "untracked files in it. Use `git add -u` or name files.")


def claude_coauthor(cmd: str):
    """The user's standing rule: no Claude attribution in this repo's commits.

    This takes precedence over any default attribution guidance, which itself
    defers to a project rule.
    """
    if re.search(r"\bgit\s+commit\b", cmd) and re.search(
            r"Co-Authored-By:\s*Claude|Generated with \[Claude Code\]|Claude-Session:", cmd):
        return ("deny", "This repo's rule is no Claude co-author or session "
                "trailers in commit messages. Drop those lines.")


def destructive(cmd: str):
    """Deletes and overwrites need an explicit OK first -- violated once."""
    if re.search(r"\brm\s+(-\w*[rf]\w*\s+)+", cmd) or re.search(r"\bgit\s+(reset\s+--hard|push\s+--force|filter-branch)", cmd):
        return ("deny", "Deletes, overwrites and history rewrites need the user's "
                "explicit OK first. Ask, naming exactly what would be removed.")


def merged_mtz_refine(cmd: str):
    """A merged MTZ must be scored on BOTH column variants.

    A healthy writer gives the same refinement from `F(+)/SigF(+)` and
    `I(+)/SigI(+)`; a gap between them is an export defect, which is how the
    float32 SigF cancellation was finally located. See docs/exporting.md.
    """
    if re.search(r"refine_(merged|june)\.sh|--write-merged-mtz", cmd) and \
            not re.search(r"\bI\(\+\)|SigI\(\+\)|for\s+V\s+in[^\n]*I", cmd):
        return ("advise", "Scoring a merged MTZ: refine BOTH F(+),SigF(+),F(-),SigF(-) "
                "and I(+),SigI(+),I(-),SigI(-). They should agree; a gap between "
                "them is an export defect, not a model result (docs/exporting.md).")


def asu_table_build(cmd: str):
    """--reassign renumbers ids a trained model indexes by."""
    if "prepare_asu_ids.py" in cmd and "--reassign" in cmd:
        return ("advise", "--reassign renumbers asu_id with a fresh ordering, so it no "
                "longer indexes the hkl_table of any model already trained on this "
                "file. Without the flag the script keeps the existing ids and only "
                "rewrites asu_id_to_hkl.pt, which is what rebuilding a lost table "
                "needs (docs/exporting.md).")


RULES = [find_on_cluster, bare_python, git_add_broad, claude_coauthor,
         destructive, merged_mtz_refine, asu_table_build]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    cmd = (payload.get("tool_input") or {}).get("command", "")
    if not cmd:
        return 0
    cmd = _strip_heredocs(cmd)
    global CWD
    CWD = Path(payload.get("cwd") or ".")

    notes = []
    for rule in RULES:
        try:
            hit = rule(cmd)
        except Exception:
            continue
        if not hit:
            continue
        kind, reason = hit
        if kind == "deny":
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }}))
            return 0
        notes.append(reason)

    if notes:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": " ".join(notes),
        }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
