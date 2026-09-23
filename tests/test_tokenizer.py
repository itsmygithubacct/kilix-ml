from kilix_ml.tokenizer import train, Tokenizer
from kilix_ml.format import example


def test_byte_roundtrip_native_parity_and_controls(tmp_path):
    corpus = ['Hello world 123 {"name":"rename_pane"}', 'a_b café 你好 🙂\\\n'] * 30
    tok, native = train(corpus, tmp_path / 'tokenizer.json', 350)
    for text in corpus + ['unknown Русский\t\x00', '<|calls|> literal', '1234567890']:
        assert tok.encode(text) == native.encode(text, add_special_tokens=False).ids if '<|calls|>' not in text else True
        assert tok.decode(tok.encode(text)) == text
        assert not set(tok.encode(text)) & set(tok.special.values())
    assert len(tok.encode('12345')) == 5
    assert Tokenizer.load(tmp_path / 'tokenizer.json').encode('Hello') == tok.encode('Hello')
    ids, labels = example(tok, 'Hello', [], [])
    start = labels.index(tok.special['<|calls|>'])
    assert all(i == -100 for i in labels[:start])
    assert ids[start:] == labels[start:]
