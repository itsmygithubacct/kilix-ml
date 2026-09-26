#!/usr/bin/env python3
"""Generate kilix-needle `apps`-job training examples from this library's corpus.

    python3 generate.py [--seed N] [--per-template K] [--exclude FILE ...] [--out FILE]

Every example is {"query": ..., "actions": [[tool, {args}], ...],
"spans": [[words, value], ...]}: the actions are the job's own five tools
(launch, show, pane_stat, game, settings) with canonical arguments, and
`spans` says which words of the query each bound value came from. The output
depends only on the corpus and the seed.

Slots: {app}, {game}, {tool}, {item}, {stat}, {section}, each optionally
numbered ({item2}) for a second, different value of the same kind. A
template's actions bind them as "$app", "$item2" and so on.

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
_KINDS = {"app": "apps.json", "game": "games.json", "tool": "tools.json", "item": "items.json",
          "stat": "stats.json", "section": "sections.json"}


def _fold(text: str) -> str:
    return " ".join(text.casefold().split())


def load_vocab(corpus: Path = CORPUS) -> dict:
    return {kind: json.loads((corpus / "vocab" / name).read_text(encoding="utf-8"))
            for kind, name in _KINDS.items()}


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
    text = _SLOT.sub(lambda m: surface[m.group(1) + m.group(2)], template["text"])

    def resolve(value):
        if isinstance(value, str) and value.startswith("$"):
            return bound[value[1:]]
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
