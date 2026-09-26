import hashlib
import json
import re
from pathlib import Path

PACK = Path(__file__).resolve().parents[1] / 'domains' / 'kilix_panes'


def locked():
    out = {}
    for line in (PACK / 'requirements-train.lock').read_text().splitlines():
        match = re.match(r'^([A-Za-z0-9_.-]+)==(\S+)', line)
        if match:
            out[re.sub(r'[-_.]+', '-', match[1]).lower()] = match[2]
    return out


def records():
    return json.loads((PACK / 'requirements-train.licences.json').read_text())


def test_every_locked_package_has_a_licence_record_at_its_version():
    data = records()
    assert data['lock_sha256'] == hashlib.sha256(
        (PACK / 'requirements-train.lock').read_bytes()).hexdigest()
    have = {r['package']: r['version'] for r in data['records']}
    assert have == locked()


def test_every_record_has_a_licence_and_its_texts():
    for record in records()['records']:
        assert record['licence'], record['package']
        assert record['texts'], record['package']
        for text in record['texts']:
            assert re.fullmatch(r'[0-9a-f]{64}', text['sha256']) and text['bytes'] > 0


def test_the_apps_pack_trains_in_the_same_environment():
    apps = PACK.parent / 'kilix_apps'
    for name in ('requirements-train.lock', 'requirements-train.in', 'requirements-train.licences.json'):
        assert (apps / name).read_bytes() == (PACK / name).read_bytes(), name
