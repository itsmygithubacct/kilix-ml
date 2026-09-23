"""A domain owns schemas and optional semantic validation; the engine is generic."""
import importlib.util
import json
from pathlib import Path


class Domain:
    def __init__(self, root):
        self.root = Path(root)
        self.tools = json.loads((self.root / 'tools.json').read_text())
        self.contract = json.loads((self.root / 'contract.json').read_text()) if (self.root / 'contract.json').exists() else None
        self._validator = None
        if (self.root / 'validate.py').exists():
            spec = importlib.util.spec_from_file_location('kilix_domain_validator', self.root / 'validate.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._validator = module.validate_many

    def validate_many(self, rows):
        if self._validator is not None:
            return self._validator(rows)
        from .grammar import compile_tools
        from .format import canonical_calls, compact
        out = []
        for row in rows:
            try:
                calls = canonical_calls(row['calls'], self.tools)
                grammar = compile_tools(self.tools, row['request'])
                state = grammar.advance(grammar.initial_state(), compact(calls).encode())
                valid = state is not None and grammar.is_complete(state)
            except (ValueError, KeyError, TypeError):
                valid = False
            out.append({'actions': [[c['name'], c['arguments']] for c in row['calls']] if valid else [],
                        'status': 0 if valid else 1, 'refusals': [] if valid else ['schema mismatch']})
        return out

