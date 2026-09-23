"""The one renderer shared by training and inference."""
import json

FORMAT = 'kilix-chat/v1'


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def canonical_calls(calls, tools):
    registry = {t['name']: t for t in tools}
    out = []
    for call in calls:
        properties = registry[call['name']]['parameters']['properties']
        args = call['arguments']
        if set(args) - set(properties):
            raise ValueError('undeclared arguments')
        out.append({'name': call['name'], 'arguments': {k: args[k] for k in properties if k in args}})
    return out


def prefix(tokenizer, tools=None, system=None):
    ids = [tokenizer.special['<bos>']]
    for role, text in [('system', system), ('tools', None if tools is None else compact(tools))]:
        if text is not None:
            ids += [tokenizer.special[f'<|{role}|>']] + tokenizer.encode(text) + [tokenizer.special['<|end|>']]
    return ids


def conversation(tokenizer, messages, tools=None):
    messages = list(messages)
    system = messages.pop(0)['content'] if messages and messages[0]['role'] == 'system' else None
    ids = prefix(tokenizer, tools, system)
    for message in messages:
        role = message['role']
        if role not in ('user', 'assistant', 'result'):
            raise ValueError(f'unsupported message role: {role}')
        ids += [tokenizer.special[f'<|{role}|>']]
        if role == 'assistant':
            if 'calls' in message:
                ids += [tokenizer.special['<|calls|>']] + tokenizer.encode(compact(message['calls']))
            else:
                ids += [tokenizer.special['<|text|>']] + tokenizer.encode(message['content'])
        else:
            ids += tokenizer.encode(message['content'])
        ids += [tokenizer.special['<|end|>']]
    return ids


def example(tokenizer, request, calls, tools, system=None):
    messages = ([{'role': 'system', 'content': system}] if system is not None else [])
    messages.append({'role': 'user', 'content': request})
    prompt = conversation(tokenizer, messages, tools) + [tokenizer.special['<|assistant|>']]
    answer = ([tokenizer.special['<|calls|>']] + tokenizer.encode(compact(canonical_calls(calls, tools)))
              + [tokenizer.special['<|end|>'], tokenizer.special['<eos>']])
    return prompt + answer, [-100] * len(prompt) + answer
