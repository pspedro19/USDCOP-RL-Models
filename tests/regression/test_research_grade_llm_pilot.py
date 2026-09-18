"""Offline/mock tests; fixtures are software controls, never thesis market evidence."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.research.llm_experiment_v2 import (
    SNAPSHOT_FIELDS,
    PilotBlocked,
    PilotRunner,
    PilotStore,
    allowed_url,
    cohort_status,
    digest,
    file_digest,
    freeze_file,
    prepare_manifest,
    prompts,
    quote_cost_micro,
    render_legacy,
    safe_path,
    validate_context,
    verify_manifest,
)

FREEZE = datetime(2026, 9, 12, 18, tzinfo=UTC)


@pytest.fixture
def prepared(tmp_path):
    sessions = []
    day = FREEZE.date() + timedelta(days=1)
    while len(sessions) < 20:
        if day.weekday() < 5:
            sessions.append({"session_date": day.isoformat(),
                             "open_utc": f"{day.isoformat()}T13:00:00+00:00",
                             "close_utc": f"{day.isoformat()}T18:00:00+00:00"})
        day += timedelta(days=1)
    calendar = {"source": "MOCK_CALENDAR_NOT_MARKET_EVIDENCE", "published_at_utc": FREEZE.isoformat(), "sessions": sessions}
    dictionary = {"features": [
        {"name": "rsi", "meaning": "mock RSI standardized only on development", "native_unit": "0..100",
         "representation": "zscore_clipped", "mean": 50, "std": 10, "clip": [-5, 5]},
        {"name": "p_regime", "meaning": "mock filtered posterior", "native_unit": "probability",
         "representation": "probability"}]}
    config = {"status": "OPERATOR_APPROVED", "variants": ["L0", "L1", "L2"],
              "cohort_sessions": 20, "bars_per_session": 59, "budget_usd": 100,
              "max_input_tokens": 32768,
              "sampling": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 256, "max_retries": 1},
              "providers": {}}
    for name in ("deepseek", "azure_openai"):
        config["providers"][name] = {"requested_model": f"MOCK_{name}", "api_version": "MOCK_API",
            "expected_served_model": "MOCK_SERVED_SNAPSHOT", "pricing_model": f"MOCK_{name}",
            "endpoint": "https://api.deepseek.com" if name == "deepseek" else "https://fixture.openai.azure.com",
            "pricing": {"model": f"MOCK_{name}", "source_url": ("https://api-docs.deepseek.com/quick_start/pricing"
                        if name == "deepseek" else "https://azure.microsoft.com/en-us/pricing/details/azure-openai/"),
                        "evidence_sha256": "a" * 64, "observed_at_utc": FREEZE.isoformat(),
                        "valid_until_utc": "2027-01-01T00:00:00+00:00",
                        "input_usd_per_million": 0.01, "output_usd_per_million": 0.01}}
    artifacts = {}
    for name in ("dataset", "schema", "scaler", "cost_contract"):
        path = tmp_path / f"{name}.json"
        path.write_text('{}', encoding="utf-8")
        artifacts[name] = path
    artifacts['schema'].write_text(json.dumps({'groups': {'macro': [], 'regimen': ['p_regime']}}), encoding='utf-8')
    artifacts['scaler'].write_text(json.dumps({'features': ['rsi'], 'mean': [50], 'scale': [10],
                                              'macro_mean': [], 'macro_scale': []}), encoding='utf-8')
    for name in ("deepseek", "azure_openai"):
        path = tmp_path / f"pricing_{name}.json"
        path.write_text('{"fixture":"MOCK_TARIFF_NOT_A_REAL_PRICE"}', encoding='utf-8')
        artifacts['pricing_' + name] = path
        config['providers'][name]['pricing']['evidence_sha256'] = file_digest(path)
    manifest = prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)
    return manifest, config, calendar, dictionary, artifacts


def context(manifest, bar=0):
    session = manifest["cohort"][0]
    opened = datetime.fromisoformat(session["open_utc"])
    cutoff = opened + timedelta(minutes=(bar + 1) * 5)
    row = {"session_date": session["session_date"], "bar": bar, "retrospective": False,
            "context_created_at_utc": cutoff.isoformat(),
            "cutoff_utc": cutoff.isoformat(), "decision_deadline_utc": (cutoff + timedelta(minutes=5)).isoformat(),
            "dataset_sha256": manifest["artifacts"]["dataset"]["sha256"], "snapshot_sha256": "d" * 64,
            "daily_features": {"p_regime": 1.0}, "daily_available_at_utc": (opened - timedelta(hours=1)).isoformat(),
            "close": 4000.0 + bar,
            "cost_context": {"unit": "decimal_return_per_abs_delta_weight", "one_way_return_per_unit": 0.001,
                             "source": "MOCK_COST_ASSUMPTION", "contract_sha256": manifest["artifacts"]["cost_contract"]["sha256"]},
            "market": [{"bar": b, "features": {"rsi": -0.2},
                        "observed_at_utc": (opened + timedelta(minutes=(b + 1) * 5)).isoformat(),
                        "received_at_utc": (opened + timedelta(minutes=(b + 1) * 5)).isoformat()}
                       for b in range(max(0, bar - 23), bar + 1)]}
    snapshot = Path(manifest['artifacts']['dataset']['path']).parent / f'live-snapshot-{bar}.json'
    snapshot.write_text(json.dumps({k: row[k] for k in SNAPSHOT_FIELDS}), encoding='utf-8')
    row['snapshot_path'] = str(snapshot)
    row['snapshot_sha256'] = file_digest(snapshot)
    return row


class MockTransport:
    def __init__(self, malformed=0):
        self.requests = []
        self.malformed = malformed

    def generate(self, provider, spec, request):
        self.requests.append(deepcopy(request))
        text = 'invalid' if len(self.requests) <= self.malformed else '{"direccion":"long","tamano":0.5,"confianza":0.7}'
        return {"content": text, "served_model": "MOCK_SERVED_SNAPSHOT", "response_id": "MOCK_ID",
                "request_id": "MOCK_REQUEST", "usage": {"prompt_tokens": 10, "completion_tokens": 8},
                "raw_sha256": "e" * 64, "system_fingerprint": "MOCK_FP"}


def test_offline_frozen_variants_only_add_dictionary_and_state(prepared):
    manifest = prepared[0]
    ctx, state = context(manifest), {"position": 0.5, "session_pnl_decimal": -0.1}
    l0, l1, l2 = [prompts(v, ctx, manifest, state) for v in ("L0", "L1", "L2")]
    assert l0[1] == render_legacy(ctx)
    assert l1[1].startswith(l0[1]) and l2[1].startswith(l1[1])
    assert l0[0] == l1[0] == l2[0]
    assert 'z=clip' in l1[1] and 'Estado PROPIO' not in l1[1]
    assert 'Estado PROPIO' in l2[1] and 'MOCK_COST_ASSUMPTION' in l2[1]


def test_cohort_first_twenty_future_dates_and_manifest_no_overwrite(prepared, tmp_path):
    manifest = prepared[0]
    assert len(manifest['cohort']) == 20
    assert manifest['cohort'][0]['session_date'] == '2026-09-14'
    path = tmp_path / 'manifest.json'
    freeze_file(path, manifest)
    with pytest.raises(FileExistsError):
        freeze_file(path, manifest)
    altered = deepcopy(manifest)
    altered['config']['sampling']['top_p'] = 1.0
    with pytest.raises(PilotBlocked, match='digest mismatch'):
        verify_manifest(altered)


@pytest.mark.parametrize('mutation', ['price_missing', 'stale_price', 'over_budget', 'past_calendar', 'invalid_dictionary'])
def test_admission_is_fail_closed(prepared, mutation):
    _, config, calendar, dictionary, artifacts = deepcopy(prepared)
    if mutation == 'price_missing':
        config['providers']['deepseek']['pricing'] = None
    elif mutation == 'stale_price':
        config['providers']['deepseek']['pricing']['valid_until_utc'] = '2025-01-01T00:00:00Z'
    elif mutation == 'over_budget':
        config['providers']['deepseek']['pricing']['input_usd_per_million'] = 1000
    elif mutation == 'past_calendar':
        calendar['sessions'] = []
    else:
        dictionary['features'][0]['std'] = 0
    with pytest.raises(PilotBlocked):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)


@pytest.mark.parametrize('field', ['future', 'retrospective', 'late_receipt', 'dataset', 'missing_bar'])
def test_context_temporal_and_identity_checks(prepared, field):
    manifest = prepared[0]
    ctx = context(manifest)
    now = datetime.fromisoformat(ctx['cutoff_utc']) + timedelta(seconds=1)
    if field == 'future':
        now -= timedelta(hours=1)
    elif field == 'retrospective':
        ctx['retrospective'] = True
    elif field == 'late_receipt':
        ctx['market'][0]['received_at_utc'] = ctx['decision_deadline_utc']
    elif field == 'dataset':
        ctx['dataset_sha256'] = 'f' * 64
    else:
        ctx['market'] = []
    with pytest.raises(PilotBlocked):
        validate_context(ctx, manifest, now)


def test_request_identity_recursive_state_idempotence_and_sidecar(prepared, tmp_path):
    manifest = prepared[0]
    store = PilotStore(tmp_path / 'shared.sqlite')
    transport = MockTransport()
    ctx = context(manifest)
    runner = PilotRunner(manifest, store, tmp_path / 'responses', transport,
                         clock=lambda: datetime.fromisoformat(ctx['cutoff_utc']) + timedelta(seconds=1))
    first = runner.decide(ctx, 'deepseek', 'L2')
    assert first['weight'] == 0.5 and first['state_after']['session_pnl_decimal'] == -0.0005
    assert first['requested_model'] == transport.requests[0]['model'] == 'MOCK_deepseek'
    assert first['sampling']['top_p'] == transport.requests[0]['top_p'] == 0.9
    assert first['served_model'] == 'MOCK_SERVED_SNAPSHOT'
    assert runner.decide(ctx, 'deepseek', 'L2') == first and len(transport.requests) == 1
    ctx = context(manifest, 1)
    second = runner.decide(ctx, 'deepseek', 'L2')
    assert second['previous_weight'] == 0.5 and second['state_before']['bars_in_position'] == 1
    assert second['state_before']['session_pnl_decimal'] == pytest.approx(-0.000375)
    assert '"position":0.5' in transport.requests[-1]['messages'][1]['content']
    sidecar = Path(second['attempts'][0]['sidecar_path'])
    assert json.loads(sidecar.read_text())['content'].startswith('{"direccion"')
    assert cohort_status(manifest, store.records(manifest))['status'] == 'INCOMPLETE_OR_INVALID'


def test_retry_budget_reserve_and_uncertain_request_no_reissue(prepared, tmp_path):
    manifest = prepared[0]
    store = PilotStore(tmp_path / 'shared.sqlite')
    store.admit(manifest)
    call_id = 'crash-before-response'
    store.reserve_attempt(manifest, call_id, 'deepseek')
    with pytest.raises(PilotBlocked, match='already started'):
        PilotStore(store.path).reserve_attempt(manifest, call_id, 'deepseek')
    ctx, transport = context(manifest), MockTransport(malformed=1)
    runner = PilotRunner(manifest, store, tmp_path / 'responses', transport,
                         clock=lambda: datetime.fromisoformat(ctx['cutoff_utc']) + timedelta(seconds=1))
    result = runner.decide(ctx, 'azure_openai', 'L0')
    assert len(result['attempts']) == 2 and len(transport.requests) == 2


def test_global_budget_cohort_admission_atomic_across_connections(prepared, tmp_path):
    manifest = prepared[0]
    # Inflate only test allocations and rehash. Models/prices are MOCK fixtures.
    candidates = []
    for index in range(2):
        item = deepcopy(manifest)
        item['allocation_micro_usd'] = 60_000_000
        item['test_cohort'] = index
        item.pop('manifest_sha256')
        item['manifest_sha256'] = digest(item)
        candidates.append(item)
    path = tmp_path / 'shared.sqlite'
    PilotStore(path)

    def admit(item):
        try:
            PilotStore(path).admit(item)
            return True
        except PilotBlocked:
            return False

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(admit, candidates)) == [False, True]


@pytest.mark.parametrize('path', ['.env', '.env.production', 'secrets/file.json', 'x/private.key',
                                   'credentials-prod.json', 'service-account-test.json'])
def test_secret_paths_rejected_before_io(path):
    with pytest.raises(PilotBlocked, match='sensitive path'):
        safe_path(Path(path))


@pytest.mark.parametrize('url', ['https://api.deepseek.com?api_key=x',
                               'https://user:pass@api.deepseek.com',
                               'https://api.deepseek.com#fragment',
                               'https://attacker.invalid', 'http://api.deepseek.com'])
def test_endpoint_rejects_secret_and_unapproved_locations(url):
    with pytest.raises(PilotBlocked):
        allowed_url(url, provider='deepseek')


def test_tariff_hash_is_bound_to_real_evidence_bytes(prepared):
    _, config, calendar, dictionary, artifacts = prepared
    artifacts['pricing_deepseek'].write_text('changed quote', encoding='utf-8')
    with pytest.raises(PilotBlocked, match='tariff evidence'):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)


def test_future_bar_cannot_be_claimed_at_earlier_cutoff(prepared):
    manifest = prepared[0]
    row = context(manifest, 58)
    early = context(manifest, 0)
    row['cutoff_utc'] = early['cutoff_utc']
    row['decision_deadline_utc'] = early['decision_deadline_utc']
    now = datetime.fromisoformat(early['cutoff_utc']) + timedelta(seconds=1)
    with pytest.raises(PilotBlocked, match='exact session-open'):
        validate_context(row, manifest, now)


def test_live_snapshot_must_be_recoverable_and_match_context(prepared):
    manifest = prepared[0]
    row = context(manifest)
    Path(row['snapshot_path']).write_text('{}', encoding='utf-8')
    now = datetime.fromisoformat(row['cutoff_utc']) + timedelta(seconds=1)
    with pytest.raises(PilotBlocked, match='snapshot content hash'):
        validate_context(row, manifest, now)


def test_exact_request_and_context_archived_before_any_network(prepared, tmp_path):
    manifest = prepared[0]
    store = PilotStore(tmp_path / 'budget.sqlite')
    ctx = context(manifest)
    base = MockTransport()

    class CheckingTransport:
        def generate(self, provider, spec, request):
            paths = list((tmp_path / 'responses').glob('*.request.json'))
            assert len(paths) == 1
            payload = json.loads(paths[0].read_text(encoding='utf-8'))
            assert payload['request'] == request and payload['context'] == ctx
            assert 'headers' not in payload and 'api_key' not in payload
            return base.generate(provider, spec, request)

    runner = PilotRunner(manifest, store, tmp_path / 'responses', CheckingTransport(),
                         clock=lambda: datetime.fromisoformat(ctx['cutoff_utc']) + timedelta(seconds=1))
    result = runner.decide(ctx, 'deepseek', 'L1')
    assert file_digest(Path(result['request_archive_path'])) == result['request_archive_sha256']


@pytest.mark.parametrize('usage', [
    {'prompt_tokens': -1, 'completion_tokens': 3},
    {'prompt_tokens': 10.5, 'completion_tokens': 3},
    {'prompt_tokens': 10, 'completion_tokens': 257},
    {'prompt_tokens': 999999, 'completion_tokens': 3},
])
def test_bad_or_excess_usage_halts_all_providers_durably(prepared, tmp_path, usage):
    manifest = prepared[0]
    store = PilotStore(tmp_path / 'budget.sqlite')
    ctx = context(manifest)

    class BadUsage(MockTransport):
        def generate(self, provider, spec, request):
            response = super().generate(provider, spec, request)
            response['usage'] = usage
            return response

    transport = BadUsage()
    runner = PilotRunner(manifest, store, tmp_path / 'responses', transport,
                         clock=lambda: datetime.fromisoformat(ctx['cutoff_utc']) + timedelta(seconds=1))
    with pytest.raises(PilotBlocked, match='HALTED'):
        runner.decide(ctx, 'deepseek', 'L0')
    with pytest.raises(PilotBlocked, match='HALTED'):
        PilotStore(store.path).admit(manifest)
    assert len(transport.requests) == 1
    assert len(store.records(manifest)) == 1


def test_served_model_change_latches_global_halt(prepared, tmp_path):
    manifest = prepared[0]
    store = PilotStore(tmp_path / 'budget.sqlite')
    ctx = context(manifest)

    class Drift(MockTransport):
        def generate(self, provider, spec, request):
            result = super().generate(provider, spec, request)
            result['served_model'] = 'MOCK_SERVED_SNAPSHOT' if len(self.requests) == 1 else 'MOCK_CHANGED_MODEL'
            return result

    runner = PilotRunner(manifest, store, tmp_path / 'responses', Drift(),
                         clock=lambda: datetime.fromisoformat(ctx['cutoff_utc']) + timedelta(seconds=1))
    runner.decide(ctx, 'azure_openai', 'L0')
    ctx = context(manifest, 1)
    with pytest.raises(PilotBlocked, match='HALTED'):
        runner.decide(ctx, 'azure_openai', 'L0')
    with pytest.raises(PilotBlocked, match='HALTED'):
        PilotStore(store.path).admit(manifest)


def test_same_sign_resize_preserves_holding_age_and_cost_basis(prepared, tmp_path):
    manifest = prepared[0]
    store = PilotStore(tmp_path / 'budget.sqlite')
    ctx = context(manifest)

    class Resize(MockTransport):
        def generate(self, provider, spec, request):
            result = super().generate(provider, spec, request)
            size = [0.5, 1.0, 0.5][len(self.requests) - 1]
            result['content'] = json.dumps({'direccion': 'long', 'tamano': size, 'confianza': 0.7})
            return result

    runner = PilotRunner(manifest, store, tmp_path / 'responses', Resize(),
                         clock=lambda: datetime.fromisoformat(ctx['cutoff_utc']) + timedelta(seconds=1))
    first = runner.decide(ctx, 'deepseek', 'L2')
    ctx = context(manifest, 1)
    increased = runner.decide(ctx, 'deepseek', 'L2')
    ctx = context(manifest, 2)
    trimmed = runner.decide(ctx, 'deepseek', 'L2')
    assert first['state_after']['entry_price'] == 4000
    assert increased['state_after']['entry_price'] == 4000.5
    assert trimmed['state_after']['entry_price'] == 4000.5
    assert trimmed['state_after']['bars_in_position'] == 2


def test_transport_passes_exact_sampling_and_disables_redirects(monkeypatch):
    import src.research.llm_experiment_v2 as pilot
    calls = []

    class MockResponse:
        @property
        def headers(self):
            return {'x-request-id': 'MOCK_REQ'}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, count):
            return json.dumps({'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}],
                               'model': 'MOCK_SERVED', 'id': 'MOCK_ID', 'usage': {}}).encode()

    class MockOpener:
        def open(self, request, timeout):
            calls.append(json.loads(request.data))
            return MockResponse()

    def opener(handler):
        assert handler.redirect_request(None, None, 302, 'Found', {}, 'https://attacker.invalid') is None
        return MockOpener()

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'MOCK_NOT_A_REAL_CREDENTIAL')
    monkeypatch.setattr(pilot, 'build_opener', opener)
    request = {'model': 'MOCK_REQUESTED', 'temperature': 0.1, 'top_p': 0.9, 'max_tokens': 256}
    response = pilot.ExplicitChatTransport().generate('deepseek', {'endpoint': 'https://api.deepseek.com'}, request)
    assert calls == [request] and response['served_model'] == 'MOCK_SERVED'


def refresh_snapshot(row):
    path = Path(row['snapshot_path'])
    path.write_text(json.dumps({k: row[k] for k in SNAPSHOT_FIELDS}), encoding='utf-8')
    row['snapshot_sha256'] = file_digest(path)


def test_realistic_receipt_after_close_before_context_is_allowed(prepared):
    manifest = prepared[0]
    row = context(manifest)
    cutoff = datetime.fromisoformat(row['cutoff_utc'])
    row['market'][0]['received_at_utc'] = (cutoff + timedelta(seconds=2)).isoformat()
    row['context_created_at_utc'] = (cutoff + timedelta(seconds=3)).isoformat()
    refresh_snapshot(row)
    validate_context(row, manifest, cutoff + timedelta(seconds=4))
    with pytest.raises(PilotBlocked, match='deadline'):
        validate_context(row, manifest, cutoff + timedelta(minutes=5))


def test_per_bar_feature_presence_and_order_are_frozen(prepared):
    manifest = prepared[0]
    row = context(manifest, 1)
    row['market'][0]['features'] = {}
    refresh_snapshot(row)
    with pytest.raises(PilotBlocked, match='feature order'):
        validate_context(row, manifest, datetime.fromisoformat(row['cutoff_utc']) + timedelta(seconds=1))


def test_dictionary_scale_must_equal_frozen_scaler(prepared):
    _, config, calendar, dictionary, artifacts = prepared
    dictionary['features'][0]['mean'] = 999
    with pytest.raises(PilotBlocked, match='scale/mean differs'):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)


def test_first_response_unexpected_model_is_rejected_and_halts(prepared, tmp_path):
    manifest = prepared[0]
    ctx = context(manifest)
    store = PilotStore(tmp_path / 'budget.sqlite')

    class Unexpected(MockTransport):
        def generate(self, provider, spec, request):
            result = super().generate(provider, spec, request)
            result['served_model'] = 'NOT_THE_FROZEN_BASE_MODEL'
            return result

    transport = Unexpected()
    runner = PilotRunner(manifest, store, tmp_path / 'responses', transport,
                         clock=lambda: datetime.fromisoformat(ctx['cutoff_utc']) + timedelta(seconds=1))
    with pytest.raises(PilotBlocked, match='HALTED'):
        runner.decide(ctx, 'deepseek', 'L0')
    assert len(transport.requests) == 1 and not store.records(manifest)[0]['eligible_before_deadline']


@pytest.mark.parametrize(('provider', 'url'), [
    ('azure_openai', 'https://api-docs.deepseek.com/quick_start/pricing'),
    ('azure_openai', 'https://developers.openai.com/api/docs/models/gpt-4o-mini'),
    ('azure_openai', 'https://openai.com/api/pricing/'),
    ('deepseek', 'https://azure.microsoft.com/en-us/pricing/details/azure-openai/'),
    ('deepseek', 'https://platform.openai.com/docs/pricing'),
])
def test_tariff_domain_must_match_the_billing_provider(prepared, provider, url):
    _, config, calendar, dictionary, artifacts = prepared
    config['providers'][provider]['pricing']['source_url'] = url
    with pytest.raises(PilotBlocked, match='provider'):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)


@pytest.mark.parametrize('provider', [None, 'deepsek', 'openai'])
@pytest.mark.parametrize('pricing', [True, False])
def test_url_requires_a_recognized_provider(provider, pricing):
    with pytest.raises(PilotBlocked, match='provider'):
        allowed_url('https://api.deepseek.com', provider=provider, pricing=pricing)


@pytest.mark.parametrize('value', [True, False, 32768.9, 32768.0, '32768', 0, -1])
def test_input_cap_is_a_strict_positive_integer(prepared, value):
    _, config, calendar, dictionary, artifacts = prepared
    config['max_input_tokens'] = value
    with pytest.raises(PilotBlocked):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)


@pytest.mark.parametrize(('field', 'value'), [
    ('cohort_sessions', 20.0), ('bars_per_session', 59.0),
    ('max_tokens', 256.0), ('max_retries', True), ('max_retries', 1.0),
])
def test_all_reservation_counts_are_integers(prepared, field, value):
    _, config, calendar, dictionary, artifacts = prepared
    target = config['sampling'] if field in ('max_tokens', 'max_retries') else config
    target[field] = value
    with pytest.raises(PilotBlocked):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)


@pytest.mark.parametrize('field', ['requested_model', 'expected_served_model', 'pricing_model', 'api_version'])
@pytest.mark.parametrize('value', [123, True, ' ', 'model\nname', {'model': 'MOCK'}])
def test_model_metadata_is_explicit_text(prepared, field, value):
    _, config, calendar, dictionary, artifacts = prepared
    spec = config['providers']['deepseek']
    spec[field] = value
    if field == 'pricing_model':
        spec['pricing']['model'] = value
    with pytest.raises(PilotBlocked):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)


def test_freeze_rejects_naive_time_instead_of_using_host_timezone(prepared):
    _, config, calendar, dictionary, artifacts = prepared
    with pytest.raises(PilotBlocked, match='timezone'):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE.replace(tzinfo=None))


@pytest.mark.parametrize(('input_tokens', 'output_tokens'), [
    (-1, 0), (0, -1), (True, 0), (0, True), (1.5, 0), (0, '1'),
])
def test_quote_rejects_non_integer_or_negative_counts(input_tokens, output_tokens):
    price = {'input_usd_per_million': 0.01, 'output_usd_per_million': 0.01}
    with pytest.raises(PilotBlocked):
        quote_cost_micro(price, input_tokens, output_tokens)


@pytest.mark.parametrize('field', ['input_usd_per_million', 'output_usd_per_million'])
@pytest.mark.parametrize('value', [-1, 0, True, None, 'NaN', 'Infinity', '-Infinity', 'bad'])
def test_quote_rejects_invalid_tariffs(field, value):
    price = {'input_usd_per_million': 0.01, 'output_usd_per_million': 0.01}
    price[field] = value
    with pytest.raises(PilotBlocked):
        quote_cost_micro(price, 1, 1)


def test_quote_is_an_exact_upper_bound_even_beyond_decimal_context_precision():
    price = {'input_usd_per_million': '1.00000000000000000000000000001',
             'output_usd_per_million': '0.00000000000000000000000000001'}
    assert quote_cost_micro(price, 1, 0) == 2
    assert quote_cost_micro(price, 0, 0) == 0


def test_existing_valid_allocation_remains_integer_and_unchanged(prepared):
    manifest = prepared[0]
    assert manifest['per_attempt_reserve_micro_usd'] == {'deepseek': 331, 'azure_openai': 331}
    assert manifest['allocation_micro_usd'] == 4_686_960
    assert type(manifest['allocation_micro_usd']) is int


def test_numeric_hash_cannot_be_coerced_into_pricing_evidence(prepared, monkeypatch):
    import src.research.llm_experiment_v2 as pilot

    _, config, calendar, dictionary, artifacts = prepared
    config['providers']['deepseek']['pricing']['evidence_sha256'] = int('1' * 64)
    reads = []

    def no_reads(path):
        reads.append(path)
        return '1' * 64

    monkeypatch.setattr(pilot, 'file_digest', no_reads)
    with pytest.raises(PilotBlocked, match='SHA256'):
        prepare_manifest(config, calendar, dictionary, artifacts, now=FREEZE)
    assert reads == []


@pytest.mark.parametrize(('provider', 'url'), [
    ('deepseek', 'https://api-docs.deepseek.com/quick_start/pricing/'),
    ('deepseek', 'https://api.deepseek.com/'),
    ('azure_openai', 'https://azure.microsoft.com/en-us/pricing/details/azure-openai/'),
    ('azure_openai', 'https://learn.microsoft.com/en-us/azure/cost-management-billing/'),
])
def test_provider_scoped_pricing_domains_are_necessary_not_price_authentication(provider, url):
    assert allowed_url(url, provider=provider, pricing=True) == url


def test_quote_does_not_depend_on_the_callers_decimal_context():
    from decimal import localcontext

    price = {'input_usd_per_million': '0.10000000000000000000000000001',
             'output_usd_per_million': '0.10000000000000000000000000001'}
    for precision in (1, 2, 28, 50):
        with localcontext() as ctx:
            ctx.prec = precision
            assert quote_cost_micro(price, 5, 5) == 2
