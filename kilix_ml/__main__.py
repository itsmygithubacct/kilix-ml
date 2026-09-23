"""Persistent JSONL inference, with optional domain validation and no execution."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .domain import Domain
from .runtime.complete import Engine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--domain', type=Path)
    p.add_argument('--request')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=192)
    a = p.parse_args()
    if a.validate and not a.domain:
        p.error('--validate requires --domain')
    domain = Domain(a.domain) if a.domain else None
    engine = Engine(a.bundle, max_new_tokens=a.max_new_tokens)
    rows = [{'request': a.request}] if a.request is not None else (json.loads(line) for line in sys.stdin if line.strip())
    for row in rows:
        try:
            messages = row.get('messages', row.get('request'))
            result = engine.complete(messages, domain.tools if domain else None)
            output = asdict(result)
            if a.validate:
                request = messages if isinstance(messages, str) else messages[-1]['content']
                output['validation'] = domain.validate_many([{'request': request, 'calls': result.calls}])[0]
            print(json.dumps(output, ensure_ascii=False), flush=True)
        except (ValueError, TypeError, KeyError) as exc:
            print(json.dumps({'error': str(exc), 'calls': []}), flush=True)


if __name__ == '__main__':
    main()
