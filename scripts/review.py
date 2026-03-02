#!/usr/bin/env python3
"""Interactive review tool for rollout classifications.

Usage:
    python scripts/review.py                     # review latest exp
    python scripts/review.py exp-3               # review specific exp
    python scripts/review.py exp-3 --env moltbook_influence
    python scripts/review.py exp-3 --filter unclear scheming
"""

import argparse
import json
import re
import sys
import termios
import tty
from pathlib import Path

DATA_DIR = Path("data/experiments")

CLASSIFICATIONS = ["honest", "scheming", "overrefusal", "unclear"]

KEYS = {
    "h": "honest",
    "s": "scheming",
    "o": "overrefusal",
    "u": "unclear",
    "k": "keep",
    "b": "back",
    "q": "quit",
}

C = {
    "honest":      "\033[92m",
    "scheming":    "\033[91m",
    "overrefusal": "\033[93m",
    "unclear":     "\033[94m",
    "bold":        "\033[1m",
    "dim":         "\033[2m",
    "header":      "\033[95m",
    "reset":       "\033[0m",
}


def color(text: str, *keys: str) -> str:
    return "".join(C[k] for k in keys) + text + C["reset"]


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _wrap_line(line: str, width: int = 86, indent: str = "  ") -> str:
    """Word-wrap a single line of text, preserving its leading whitespace."""
    # Detect list items (-, *, 1., 2., etc.) and extra-indent continuation
    stripped = line.lstrip()
    lead = line[: len(line) - len(stripped)]  # original indentation
    prefix = indent + lead

    # Check for list/numbered item so continuation lines are indented further
    m = re.match(r"^(\s*(?:\d+[.)]\s+|[-*•]\s+))", line)
    cont_prefix = prefix + "    " if m else prefix

    words = stripped.split()
    if not words:
        return ""

    lines = []
    current = prefix + words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = cont_prefix + word
    lines.append(current)
    return "\n".join(lines)


def fmt(text: str, width: int = 86) -> str:
    """Format multi-paragraph text for terminal display.

    Preserves:
    - Paragraph breaks (blank lines)
    - List items (-, *, numbered)
    - Existing indentation structure
    Word-wraps long lines.
    """
    output = []
    for paragraph in text.split("\n"):
        stripped = paragraph.rstrip()
        if stripped == "":
            output.append("")  # blank line preserved
        else:
            output.append(_wrap_line(stripped, width))
    # Collapse 3+ consecutive blank lines to 2
    result = []
    blanks = 0
    for line in output:
        if line == "":
            blanks += 1
            if blanks <= 2:
                result.append("")
        else:
            blanks = 0
            result.append(line)
    return "\n".join(result).strip()


# ------------------------------------------------------------------
# File I/O
# ------------------------------------------------------------------

def load_env_review(review_path: Path) -> list:
    entries = []
    with open(review_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_env_review(review_path: Path, entries: list):
    with open(review_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Experiment discovery
# ------------------------------------------------------------------

def resolve_experiment(arg: str | None) -> str:
    """Return experiment ID from arg or latest exp-N."""
    if arg:
        return arg
    # Find latest
    nums = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and d.name.startswith("exp-"):
            try:
                nums.append(int(d.name[4:]))
            except ValueError:
                pass
    if not nums:
        print("No experiments found in data/experiments/")
        sys.exit(1)
    return f"exp-{max(nums)}"


def list_env_dirs(experiment_id: str, subdir: str | None = None) -> list:
    exp_path = DATA_DIR / experiment_id
    if subdir:
        exp_path = exp_path / subdir
    if not exp_path.exists():
        return []
    return sorted(
        d for d in exp_path.iterdir()
        if d.is_dir() and (d / "review.jsonl").exists()
    )


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

def _section(title: str, body: str):
    """Print a labeled section with formatted body text."""
    print(color(title, "bold"))
    print(color("─" * 90, "dim"))
    print(fmt(body))
    print()


def print_rollout(entry: dict, pos: int, total: int, env_name: str):
    print("\033[2J\033[H", end="")  # clear screen

    conf = entry.get("metadata", {}).get("monitor_confidence")
    conf_str = f"{conf:.2f}" if conf is not None else "manual"
    cls = entry["classification"]

    # Header bar
    print(color(
        f" {env_name}  │  {pos + 1} / {total}  │  monitor confidence: {conf_str} ",
        "header", "bold",
    ))
    print()

    _section("TASK", entry.get("instruction", ""))

    thought = entry.get("thought", "").strip()
    if thought:
        _section("CHAIN OF THOUGHT", thought)

    _section("RESPONSE", entry.get("output", ""))

    cls_color = cls if cls in C else "reset"
    print(f"  Classification:  {color(cls.upper(), cls_color, 'bold')}")
    print()

    print(color("─" * 90, "dim"))
    print(
        f"  {color('[H]', 'bold')}onest  "
        f"{color('[S]', 'bold')}cheming  "
        f"{color('[O]', 'bold')}verrefusal  "
        f"{color('[U]', 'bold')}nclear  "
        f"  {color('[K]keep', 'dim')}  "
        f"{color('[B]back', 'dim')}  "
        f"{color('[Q]quit', 'dim')}"
    )
    print("  > ", end="", flush=True)


# ------------------------------------------------------------------
# Review loop for a single env
# ------------------------------------------------------------------

def review_env(review_path: Path, filter_cls: list | None) -> dict:
    """
    Run the interactive review loop for one env's review.jsonl.

    Returns: {"changed": int, "reviewed": int, "quit": bool}
    """
    entries = load_env_review(review_path)
    env_name = review_path.parent.name

    if filter_cls:
        indices = [i for i, e in enumerate(entries) if e["classification"] in filter_cls]
    else:
        indices = list(range(len(entries)))

    if not indices:
        return {"changed": 0, "reviewed": 0, "quit": False}

    total = len(indices)
    pos = 0
    changed = 0
    quit_requested = False

    while 0 <= pos < total:
        idx = indices[pos]
        entry = entries[idx]
        print_rollout(entry, pos, total, env_name)

        key = getch().lower()
        print(key)

        if key == "q":
            quit_requested = True
            break
        elif key == "b":
            pos = max(0, pos - 1)
        elif key in ("k", "\r", "\n"):
            pos += 1
        elif key in KEYS and KEYS[key] not in ("keep", "back", "quit"):
            new_cls = KEYS[key]
            if new_cls != entry["classification"]:
                entry["classification"] = new_cls
                entry.setdefault("metadata", {})["monitor_confidence"] = None
                changed += 1
            pos += 1

    if changed > 0:
        save_env_review(review_path, entries)

    return {"changed": changed, "reviewed": pos, "quit": quit_requested}


# ------------------------------------------------------------------
# Summary screen
# ------------------------------------------------------------------

def print_summary(experiment_id: str, env_results: dict):
    print("\033[2J\033[H", end="")
    print(color(f" Review complete — {experiment_id}", "bold"))
    print()

    total_changed = 0
    for env_name, stats in env_results.items():
        print(f"  {color(env_name, 'bold')}: reviewed {stats['reviewed']}, changed {stats['changed']}")
        total_changed += stats["changed"]

    print()
    if total_changed > 0:
        print(f"  {total_changed} classification(s) changed and saved.")
    else:
        print("  No changes made.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive rollout reviewer")
    parser.add_argument("experiment", nargs="?", help="Experiment ID (e.g. exp-3). Defaults to latest.")
    parser.add_argument("--env", help="Only review a specific environment subfolder")
    parser.add_argument("--subdir", help="Phase subdir to look in (e.g. 'test'). Defaults to collect-phase (top level).")
    parser.add_argument(
        "--filter", nargs="+", metavar="CLASS",
        choices=CLASSIFICATIONS,
        help="Only show rollouts with these classifications",
    )
    args = parser.parse_args()

    experiment_id = resolve_experiment(args.experiment)
    exp_path = DATA_DIR / experiment_id

    if not exp_path.exists():
        print(f"Experiment not found: {exp_path}")
        sys.exit(1)

    base_path = exp_path / args.subdir if args.subdir else exp_path

    if args.env:
        env_dirs = [base_path / args.env]
        if not (env_dirs[0] / "review.jsonl").exists():
            print(f"No review.jsonl found for env '{args.env}' in {experiment_id}/{args.subdir or ''}")
            sys.exit(1)
    else:
        env_dirs = list_env_dirs(experiment_id, args.subdir)

    if not env_dirs:
        print(f"No review.jsonl files found in {experiment_id}/")
        sys.exit(1)

    print(color(f"Reviewing {experiment_id}", "bold"))
    for d in env_dirs:
        n = sum(1 for _ in open(d / "review.jsonl"))
        print(f"  {d.name}: {n} rollouts")
    print()
    input("Press Enter to start...")

    env_results = {}
    for env_dir in env_dirs:
        review_path = env_dir / "review.jsonl"
        stats = review_env(review_path, args.filter)
        env_results[env_dir.name] = stats
        if stats["quit"]:
            break

    print_summary(experiment_id, env_results)


if __name__ == "__main__":
    main()
