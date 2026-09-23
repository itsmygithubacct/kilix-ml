"""Thompson byte NFA and tokenizer mask for constrained JSON decoding."""
from functools import lru_cache


class _NFA:
    def __init__(self):
        self.edges = []
        self.eps = []
        self.start = self.node()
        self.accept = self.node()

    def node(self):
        self.edges.append({})
        self.eps.append(set())
        return len(self.edges) - 1

    def edge(self, a, b, byte):
        self.edges[a].setdefault(byte, set()).add(b)

    def epsilon(self, a, b):
        self.eps[a].add(b)

    def literal(self, data):
        a = self.node()
        cur = a
        for byte in data:
            nxt = self.node()
            self.edge(cur, nxt, byte)
            cur = nxt
        return a, cur

    def charset(self, source, target, values):
        for byte in values:
            self.edge(source, target, byte)

    def alt(self, fragments):
        a, b = self.node(), self.node()
        for x, y in fragments:
            self.epsilon(a, x)
            self.epsilon(y, b)
        return a, b

    def concat(self, fragments):
        if not fragments:
            a, b = self.node(), self.node()
            self.epsilon(a, b)
            return a, b
        for (_, end), (start, _) in zip(fragments, fragments[1:]):
            self.epsilon(end, start)
        return fragments[0][0], fragments[-1][1]

    def optional(self, frag):
        a, b = self.node(), self.node()
        self.epsilon(a, b)
        self.epsilon(a, frag[0])
        self.epsilon(frag[1], b)
        return a, b

    def star(self, frag):
        a, b = self.node(), self.node()
        self.epsilon(a, b)
        self.epsilon(a, frag[0])
        self.epsilon(frag[1], frag[0])
        self.epsilon(frag[1], b)
        return a, b

    def closure(self, states):
        out = set(states)
        todo = list(out)
        while todo:
            s = todo.pop()
            for t in self.eps[s] - out:
                out.add(t)
                todo.append(t)
        return frozenset(out)


class ByteGrammar:
    def __init__(self, nfa, call_ends=(), call_commas=(), max_calls=32):
        self.nfa = nfa
        self.call_ends = frozenset(call_ends)
        self.call_commas = frozenset(call_commas)
        self.max_calls = max_calls
        self.start = (nfa.closure((nfa.start,)), 0)

    def initial_state(self):
        return self.start

    def advance(self, state, data):
        active, count = state
        for byte in bytes(data):
            nxt = set()
            for node in active:
                if node in self.call_commas and byte == ord(",") and count >= self.max_calls:
                    continue
                nxt.update(self.nfa.edges[node].get(byte, ()))
            if not nxt:
                return None
            active = self.nfa.closure(nxt)
            if self.call_ends & active:
                count += 1
                if count > self.max_calls:
                    return None
        return active, count

    def allowed_bytes(self, state):
        active, count = state
        result = {byte for node in active for byte in self.nfa.edges[node]}
        if count >= self.max_calls:
            if self.call_commas & active:
                result.discard(ord(","))
        return result

    def is_complete(self, state):
        active, _ = state
        return self.nfa.accept in active


class TokenTrie:
    """Immutable vocabulary structure shared across request grammars."""
    def __init__(self, token_bytes):
        self.token_bytes = [bytes(x) for x in token_bytes]
        self.children = [{}]
        self.leaves = [[]]
        for tid, token in enumerate(self.token_bytes):
            if not token:
                continue
            node = 0
            for byte in token:
                nxt = self.children[node].get(byte)
                if nxt is None:
                    nxt = len(self.children)
                    self.children[node][byte] = nxt
                    self.children.append({})
                    self.leaves.append([])
                node = nxt
            self.leaves[node].append(tid)


class TokenMask:
    """Mask token ids by walking a tokenizer byte trie against the grammar NFA."""
    def __init__(self, grammar, token_bytes=None, *, trie=None):
        self.grammar = grammar
        self.trie = trie if trie is not None else TokenTrie(token_bytes)
        self.token_bytes = self.trie.token_bytes
        self.children, self.leaves = self.trie.children, self.trie.leaves
        self.cache = {}

    def allowed(self, state):
        if state in self.cache:
            return self.cache[state]
        out, todo = [], [(0, state)]
        while todo:
            node, nfa_state = todo.pop()
            out.extend(self.leaves[node])
            for byte, child in self.children[node].items():
                advanced = self.grammar.advance(nfa_state, bytes((byte,)))
                if advanced is not None:
                    todo.append((child, advanced))
        result = sorted(out)
        if len(self.cache) >= 128:
            self.cache.pop(next(iter(self.cache)))
        self.cache[state] = result
        return result

    def advance(self, state, token_id):
        token = self.token_bytes[token_id]
        if not token:
            return None
        return self.grammar.advance(state, token)
