#!/usr/bin/env python3
"""Generate kilix-needle `agents`-job training examples from this library's corpus.

    python3 generate.py [--seed N] [--per-template K] [--exclude FILE ...] [--out FILE]

Every example is {"query": ..., "actions": [[tool, {args}], ...],
"spans": [[words, value], ...]}: the actions are the job's own three tools
(agent, wait, tell), and `spans` says which words of the query each bound
value came from. The output depends only on the corpus and the seed.

Slots: {agent}, {place} bind a canonical value (the vocab file maps each to
its surface forms). {dir}, {model} bind the words as said (the checks
resolve them). {task}, {message}, {session} are payloads: lists of whole
phrases, bound exactly as inserted, because the checks require a prompt or
message verbatim. Any slot may be numbered ({dir2}) for a second, different
value. A template's actions bind "$agent", "$dir2" and so on, and a value
may combine them: "$agent@$dir" is a session reference.

`--exclude` takes kilix-needle eval files (JSONL with a "request" field):
any generated query equal to an eval request, after case and whitespace
folding, is dropped, and the count is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
_SLOT = re.compile(r"\{([a-z]+)(\d?)\}")
_KINDS = {"agent": "agents.json", "dir": "dirs.json", "place": "places.json",
          "model": "models.json", "task": "tasks.json", "message": "messages.json",
          "session": "sessions.json"}
# These bind the words as said, not a canonical key.
_AS_SAID = frozenset({"dir", "model", "task", "message", "session"})
_TOKEN = re.compile(r"\$([a-z]+\d?)")


def _fold(text: str) -> str:
    return " ".join(text.casefold().split())


def load_vocab(corpus: Path = CORPUS) -> dict:
    vocab = {}
    for kind, name in _KINDS.items():
        data = json.loads((corpus / "vocab" / name).read_text(encoding="utf-8"))
        # A payload list is a table whose surfaces are the values themselves.
        vocab[kind] = {item: [item] for item in data} if isinstance(data, list) else data
    return vocab


def load_templates(corpus: Path = CORPUS) -> list[dict]:
    templates = []
    for path in sorted((corpus / "actions").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["templates"]:
            templates.append({"text": item["text"], "actions": item["actions"],
                              "source": path.name})
    return templates


def _fill(template: dict, vocab: dict, rng: random.Random) -> dict:
    bound, surface = {}, {}
    for kind, number in dict.fromkeys(_SLOT.findall(template["text"])):
        if kind not in vocab:
            raise ValueError(f"{template['source']}: unknown slot {{{kind}{number}}}")
        key = kind + number
        taken = {value for name, value in bound.items() if name.rstrip("0123456789") == kind}
        choices = [c for c in sorted(vocab[kind]) if c not in taken]
        bound[key] = rng.choice(choices)
        surface[key] = rng.choice(vocab[kind][bound[key]])
        if kind in _AS_SAID:
            bound[key] = surface[key]
    text = _SLOT.sub(lambda m: surface[m.group(1) + m.group(2)], template["text"])

    def resolve(value):
        if isinstance(value, str) and "$" in value:
            if _TOKEN.fullmatch(value):
                return bound[value[1:]]
            return _TOKEN.sub(lambda m: str(bound[m.group(1)]), value)
        return value

    actions = [[tool, {name: resolve(value) for name, value in args.items()}]
               for tool, args in template["actions"]]
    spans = [[surface[key], bound[key]] for key in bound]
    return {"query": text, "actions": actions, "spans": spans}


def generate(seed: int = 0, per_template: int = 6, exclude: set[str] | None = None,
             corpus: Path = CORPUS) -> tuple[list[dict], int]:
    """Examples, and how many were dropped for matching an excluded request."""
    rng = random.Random(seed)
    vocab = load_vocab(corpus)
    seen, examples, excluded = set(), [], 0
    for template in load_templates(corpus):
        attempts = per_template if _SLOT.search(template["text"]) else 1
        for _ in range(attempts * 4):
            if attempts == 0:
                break
            example = _fill(template, vocab, rng)
            key = _fold(example["query"])
            if key in seen:
                continue
            seen.add(key)
            if exclude and key in exclude:
                excluded += 1
                continue
            examples.append(example)
            attempts -= 1
    rng.shuffle(examples)
    return examples, excluded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-template", type=int, default=6)
    parser.add_argument("--exclude", nargs="*", default=[], metavar="FILE")
    parser.add_argument("--out", default="-")
    args = parser.parse_args(argv)
    exclude = set()
    for path in args.exclude:
        with open(path, encoding="utf-8") as handle:
            exclude |= {_fold(json.loads(line)["request"]) for line in handle if line.strip()}
    examples, dropped = generate(args.seed, args.per_template, exclude)
    text = "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
    print(f"{len(examples)} examples, {dropped} dropped as eval requests", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
