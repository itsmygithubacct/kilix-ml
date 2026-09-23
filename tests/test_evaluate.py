from pathlib import Path
from types import SimpleNamespace

import pytest

from kilix_ml.evaluate import score


class FakeDomain:
    tools = []

    def __init__(self, name='kilix_panes'):
        self.root = Path('/synthetic/domains') / name

    def validate_many(self, rows):
        results = []
        for row in rows:
            calls = row['calls']
            results.append({
                'actions': [[call['name'], call.get('arguments', {})] for call in calls],
                'status': 0,
            })
        return results


class FakeEngine:
    def __init__(self, replies):
        self.replies = replies

    def complete(self, request, tools):
        calls, finished = self.replies[request]
        return SimpleNamespace(calls=calls, metrics={
            'finished': finished, 'prefix_cache_hit': True, 'total_ms': 100.0,
        })


def call(tool_name, **arguments):
    return {'name': tool_name, 'arguments': arguments}


def case(request, expected, tag='all', tags=None):
    value = {'id': request, 'request': request, 'calls': expected}
    if tags is not None:
        value['tags'] = tags
    else:
        value['tag'] = tag
    return value


def _score(cases, predictions, *, domain='kilix_panes', runs=1, **kwargs):
    return score(FakeEngine(predictions), FakeDomain(domain), cases, runs,
                 cases_sha256='cases-hash', bundle_manifest_sha256='bundle-hash', **kwargs)


def test_required_action_accuracy_is_per_action_and_exactly_normalized():
    cases, predictions = [], {}
    for name, expected in [
        ('rename_tab', call('rename_tab', name='Alpha')),
        ('run_in_pane', call('run_in_pane', pane='current', command='make')),
        ('open_pane_with_program', call('open_pane', side='right', program='vim')),
    ]:
        for i in range(10):
            request = f'{name}-{i}'
            cases.append(case(request, [expected], tag=name))
            predictions[request] = ([expected] if i < 7 else [], True)
    result = _score(cases, predictions)
    assert result['required_action_accuracy']['rename_tab']['accuracy'] == 0.7
    assert result['required_action_accuracy']['run_in_pane']['accuracy'] == 0.7
    assert result['required_action_accuracy']['open_pane_with_program']['accuracy'] == 0.7
    assert result['gates']['R4_required_action_accuracy_70pct'] is True


def test_empty_required_action_subsets_remain_pending():
    expected = call('rename_tab', name='Alpha')
    cases = [case('rename', [expected])]
    result = _score(cases, {'rename': ([expected], True)})
    assert result['required_action_accuracy']['rename_tab']['status'] == 'pass'
    assert result['required_action_accuracy']['run_in_pane']['status'] == 'pending'
    assert result['gates']['R4_required_action_accuracy_70pct'] is None
    assert 'R4_required_action_accuracy_70pct' in result['pending_gates']
    assert result['promotion_ready'] is False
    poor = _score([case('rename', [expected])], {'rename': ([], True)})
    assert poor['gates']['R4_required_action_accuracy_70pct'] is False


def test_baseline_regression_normalizes_different_run_counts():
    cases, expected_predictions, current_predictions = [], {}, {}
    expected = call('rename_tab', name='Alpha')
    for i in range(10):
        request = f'case-{i}'
        cases.append(case(request, [expected], tag='rename'))
        expected_predictions[request] = ([expected], True)
        current_predictions[request] = ([expected] if i < 8 else [], True)
    baseline = _score(cases, expected_predictions, runs=3)
    current = _score(cases, current_predictions, baseline=baseline)
    assert current['gates']['R3_baseline_no_gt_2_case_regression'] is True
    assert current['baseline_comparison']['per_tag']['rename']['regression_cases'] == 2.0
    too_low = dict(current_predictions)
    for i in range(7, 10):
        too_low[f'case-{i}'] = ([], True)
    failed = _score(cases, too_low, baseline=baseline)
    assert failed['gates']['R3_baseline_no_gt_2_case_regression'] is False
    with pytest.raises(ValueError, match='cases_sha256'):
        _score(cases, current_predictions, baseline={**baseline, 'cases_sha256': 'other'})


def test_second_process_hashes_and_case_identity():
    expected = call('rename_tab', name='Alpha')
    cases = [case('rename', [expected])]
    predictions = {'rename': ([expected], True)}
    first = _score(cases, predictions, runs=2)
    repeated = _score(cases, predictions, runs=2, compare_process=first)
    assert repeated['gates']['R7_within_process'] is True
    assert repeated['gates']['R7_separate_process'] is True
    changed = {**first, 'prediction_sha256': ['bad', 'bad']}
    mismatch = _score(cases, predictions, runs=2, compare_process=changed)
    assert mismatch['gates']['R7_separate_process'] is False
    unfinished_process = _score(cases, {'rename': ([expected], False)}, runs=2)
    unfinished = _score(cases, predictions, runs=2, compare_process=unfinished_process)
    assert unfinished['gates']['R7_separate_process'] is False
    with pytest.raises(ValueError, match='cases_sha256'):
        _score(cases, predictions, runs=2,
               compare_process={**first, 'cases_sha256': 'wrong'})


def test_r10_applies_only_to_synthetic_home_and_requires_both_thresholds():
    expected = call('organize_room', room='office')
    cases, predictions = [], {}
    for i in range(10):
        request = f'home-{i}'
        cases.append(case(request, [expected], tag='home'))
        predictions[request] = ([expected] if i < 9 else [], True)
    result = _score(cases, predictions, domain='synthetic_home')
    assert result['second_domain']['exact_fraction'] == 0.9
    assert result['second_domain']['grammar_complete_fraction'] == 1.0
    assert result['gates']['R10_synthetic_home'] is True
    kilix = _score(cases, predictions)
    assert kilix['gates']['R10_synthetic_home'] is None
    assert kilix['gate_status']['R10_synthetic_home'] == 'pending'

    incomplete = dict(predictions)
    incomplete['home-0'] = ([expected], False)
    result = _score(cases, incomplete, domain='synthetic_home')
    assert result['second_domain']['grammar_complete_fraction'] == 0.9
    assert result['gates']['R10_synthetic_home'] is False


def test_duplicate_risky_action_counts_as_unsafe_and_string_tags_are_atomic():
    expected = call('close_pane', pane='current')
    cases = [case('close', [expected], tags='safety')]
    result = _score(cases, {'close': ([expected, expected], True)})
    assert result['totals']['unsafe'] == 1
    assert set(result['tags']) == {'safety'}
    assert result['promotion_status'] == 'blocked'
