from pathlib import Path
import copy
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.exceptions import ConvergenceWarning

from enso_fprm.data import VARIABLES, load_enso, prepare_assets
from enso_fprm.embedding import delay_embedding
from enso_fprm.experiment import aggregate, cached_record, cases, load_config, run_case
from enso_fprm.io import PROJECT, PROC, canonical_hash, digest, write_json
from enso_fprm.masks import make_mask
from enso_fprm.metrics import evaluate
from enso_fprm.models import ALL_METHODS, GPSettings, ModelFailure, fit_predict, recover, recover_bundle, standardize
from enso_fprm.statistics import paired_bootstrap, seed_overall


@pytest.fixture
def sample():
    x = np.sin(np.arange(80) * 0.13) + np.arange(80) * 0.002
    truth = np.cos(np.arange(68) * 0.13)
    observed = truth.copy()
    observed[np.arange(68) % 3 != 0] = np.nan
    return x, truth, observed, np.arange(68)


@pytest.fixture
def fast_gp():
    return GPSettings(optimizer=False, restarts=0)


def test_author_forward_embedding_and_lengths():
    x = np.arange(422)
    a, idx = delay_embedding(x, 3, 3)
    assert a.shape == (416, 3)
    np.testing.assert_array_equal(a[0], [0, 3, 6])
    np.testing.assert_array_equal(a[-1], [415, 418, 421])
    assert delay_embedding(x, 3, 6)[0].shape == (410, 3)
    for tau in (1, 3, 6):
        assert delay_embedding(x, 3, tau, np.arange(410))[0].shape == (410, 3)


@pytest.mark.parametrize('indices', [[0, 0], [2, 1], [-1, 0], [416], [0.5]])
def test_invalid_original_indices(indices):
    with pytest.raises(ValueError):
        delay_embedding(np.arange(422), 3, 3, np.array(indices))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint32, np.uint64])
def test_unsigned_indices_embed_and_recover_like_signed_indices(dtype, fast_gp):
    reference = np.arange(12, dtype=float)
    idx = np.array([0, 1, 3, 4], dtype=dtype)
    embedded, original = delay_embedding(reference, 3, 3, idx)
    np.testing.assert_array_equal(embedded, [[0, 3, 6], [1, 4, 7], [3, 6, 9], [4, 7, 10]])
    np.testing.assert_array_equal(original, [0, 1, 3, 4])
    assert original.dtype == np.dtype(np.intp)
    np.testing.assert_array_equal(idx, np.array([0, 1, 3, 4], dtype=dtype))
    observed = np.array([0., np.nan, np.nan, 4.])
    for method in ('linear', 'fprm'):
        unsigned = recover(reference, observed, idx, method=method, settings=fast_gp)
        signed = recover(reference, observed, idx.astype(np.intp), method=method, settings=fast_gp)
        assert unsigned.error is None
        np.testing.assert_array_equal(unsigned.recovered, signed.recovered)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint32, np.uint64])
@pytest.mark.parametrize('indices', [[0, 3, 1, 4], [0, 1, 1, 4]])
def test_unsigned_indices_cannot_bypass_order_validation(dtype, indices, fast_gp):
    with pytest.raises(ValueError, match='unique and strictly increasing'):
        recover(np.arange(12), [0., np.nan, np.nan, 4.], np.array(indices, dtype=dtype),
                method='linear', settings=fast_gp)


def test_uint64_out_of_range_rejected_before_signed_conversion():
    with pytest.raises(ValueError, match='outside valid embedding range'):
        delay_embedding(np.arange(12), indices=np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64))


def test_reference_must_be_complete():
    x = np.arange(20, dtype=float)
    x[0] = np.nan
    with pytest.raises(ValueError, match='fully observed'):
        delay_embedding(x)


def test_422_by_4_real_data():
    frame = load_enso(prepare_assets())
    assert frame.shape == (422, 4)
    assert frame.index[0] == pd.Period('1990-01', freq='M')
    assert frame.index[-1] == pd.Period('2025-02', freq='M')
    assert np.isfinite(frame.to_numpy()).all()


def test_random_masks_nested_and_references_independent():
    last = np.zeros(410, bool)
    for rate in (0.1, 0.3, 0.5, 0.7, 0.9):
        mask, meta = make_mask(410, rate, 'random', 'NINO4', 0)
        assert mask.sum() == round(410 * rate)
        assert np.all(mask[last])
        again, _ = make_mask(410, rate, 'random', 'NINO4', 0)
        np.testing.assert_array_equal(mask, again)
        last = mask


def test_block_gaps_include_both_boundaries():
    starts = set()
    for seed in range(300):
        mask, meta = make_mask(20, 0.9, 'block', 'NINO4', seed)
        positions = np.flatnonzero(mask)
        assert mask.sum() == 18 and np.all(np.diff(positions) == 1)
        starts.add(meta['block_start'])
    assert {0, 2}.issubset(starts)


def test_hidden_only_metrics_hand_calculation():
    result = evaluate([1, 2, 1000], [2, 4, -1000], [True, True, False])
    assert result['rmse'] == pytest.approx(np.sqrt(2.5))
    assert result['mae'] == 1.5
    assert result['nrmse'] == pytest.approx(np.sqrt(5))
    assert result['rho'] == pytest.approx(1)
    const = evaluate([2, 2], [3, 3], [True, True])
    assert const['rho'] is None and const['nrmse'] is None
    assert set(const['undefined']) == {'rho', 'nrmse'}
    # std of many identical floats can be tiny but nonzero due to roundoff.
    mean_fill = evaluate(np.arange(205), np.full(205, 0.102), np.ones(205, bool))
    assert mean_fill['rho'] is None


def test_zero_std_and_sample_std():
    z, mean, std = standardize(np.array([[2., 2.], [4., 2.]]))
    np.testing.assert_allclose(std, [np.sqrt(2), 1])
    np.testing.assert_allclose(z[:, 1], 0)


def test_models_preserve_observations_and_equal_weights(sample, fast_gp):
    x, truth, observed, idx = sample
    results = recover_bundle(x, observed, idx, settings=fast_gp, seed=3)
    for result in results.values():
        assert result.error is None
        np.testing.assert_array_equal(result.recovered[np.isfinite(observed)], observed[np.isfinite(observed)])
        assert np.isfinite(result.recovered).all()
    assert results['multiscale_equal'].metadata['weights'] == [1/3] * 3
    hidden = np.isnan(observed)
    expected = np.mean([results[m].recovered[hidden] for m in ('fprm_tau1', 'fprm', 'fprm_tau6')], axis=0)
    np.testing.assert_allclose(results['multiscale_equal'].recovered[hidden], expected)
    weights = results['multiscale_weighted'].metadata['weights']
    assert np.all(np.array(weights) >= 0) and sum(weights) == pytest.approx(1)
    meta = results['multiscale_weighted'].metadata
    inverse = 1 / (np.array(meta['oof_mse']) + meta['epsilon'])
    np.testing.assert_allclose(weights, inverse / inverse.sum())


def test_fold_scaling_and_target_visibility(sample, fast_gp):
    x, _, observed, idx = sample
    result = recover_bundle(x, observed, idx, settings=fast_gp, methods=['multiscale_weighted'])['multiscale_weighted']
    for fold in result.metadata['folds']:
        train = np.array(fold['train_indices'])
        held = np.array(fold['validation_indices'])
        assert not set(train) & set(held)
        assert np.isfinite(observed[train]).all() and np.isfinite(observed[held]).all()
        features, _ = delay_embedding(x, 3, fold['delay'], idx)
        np.testing.assert_allclose(fold['x_mean'], features[train].mean(axis=0))
        np.testing.assert_allclose(fold['x_std'], features[train].std(axis=0, ddof=1))
        assert fold['y_mean'] == pytest.approx(observed[train].mean())


def test_no_hidden_truth_leakage_and_determinism(sample, fast_gp):
    x, truth, observed, idx = sample
    altered_truth = truth.copy()
    altered_truth[np.isnan(observed)] += 10000
    second_input = altered_truth.copy()
    second_input[np.isnan(observed)] = np.nan
    a = recover_bundle(x, observed, idx, settings=fast_gp, methods=['multiscale_weighted'], seed=8)['multiscale_weighted']
    b = recover_bundle(x, second_input, idx, settings=fast_gp, methods=['multiscale_weighted'], seed=8)['multiscale_weighted']
    np.testing.assert_array_equal(a.recovered, b.recovered)
    assert a.metadata['weights'] == b.metadata['weights']
    assert evaluate(truth, a.recovered, np.isnan(observed))['rmse'] != evaluate(altered_truth, b.recovered, np.isnan(observed))['rmse']


def test_90_percent_missing_supports_three_folds(fast_gp):
    x = np.sin(np.arange(422) * 0.05)
    y = np.cos(np.arange(410) * 0.05)
    mask, _ = make_mask(410, 0.9, 'random', 'NINO4', 0)
    y[mask] = np.nan
    result = recover_bundle(x, y, np.arange(410), settings=fast_gp, methods=['multiscale_weighted'])['multiscale_weighted']
    assert result.error is None
    assert len(result.metadata['folds']) == 9
    assert all(f['n_train'] >= 27 for f in result.metadata['folds'])


def test_linear_uses_original_month_indices(fast_gp):
    idx = np.array([0, 1, 10, 20])
    result = recover_bundle(np.arange(40), np.array([np.nan, 1., np.nan, 20.]), idx,
                            settings=fast_gp, methods=['linear'])['linear']
    np.testing.assert_allclose(result.recovered, [1, 1, 10, 20])


def test_factorization_retry_and_failure(monkeypatch, fast_gp):
    from enso_fprm import models
    original = models.GaussianProcessRegressor.fit
    calls = []
    def fail_first(self, x, y):
        calls.append(self.alpha)
        if self.alpha < 1e-7:
            raise np.linalg.LinAlgError('controlled factorization failure')
        return original(self, x, y)
    monkeypatch.setattr(models.GaussianProcessRegressor, 'fit', fail_first)
    _, _, _, meta = fit_predict(np.arange(5)[:, None], np.arange(5), [[0]], fast_gp, 1)
    assert calls == [1e-8, 1e-7] and meta['alpha'] == 1e-7
    def fail_all(self, x, y):
        raise np.linalg.LinAlgError('controlled persistent failure')
    monkeypatch.setattr(models.GaussianProcessRegressor, 'fit', fail_all)
    with pytest.raises(ModelFailure, match='all jitter'):
        fit_predict(np.arange(5)[:, None], np.arange(5), [[0]], fast_gp, 1)


def test_convergence_warning_is_retained(monkeypatch, fast_gp):
    from enso_fprm import models
    original = models.GaussianProcessRegressor.fit
    def warning_fit(self, x, y):
        warnings.warn('controlled optimizer warning', ConvergenceWarning)
        return original(self, x, y)
    monkeypatch.setattr(models.GaussianProcessRegressor, 'fit', warning_fit)
    pred, _, _, meta = fit_predict(np.arange(5)[:, None], np.arange(5), [[0]], fast_gp, 1)
    assert np.isfinite(pred).all() and meta['has_convergence_warning']


def test_scale_failure_is_not_silently_dropped(monkeypatch, sample, fast_gp):
    from enso_fprm import models
    original = models.fit_predict
    def one_scale_fails(x, y, pred, settings, seed):
        if x.shape[1] == 3 and np.allclose(x[:, 1] - x[:, 0], 3):
            raise ModelFailure('controlled tau=3 failure')
        return original(x, y, pred, settings, seed)
    monkeypatch.setattr(models, 'fit_predict', one_scale_fails)
    y = np.arange(68, dtype=float)
    y[::2] = np.nan
    results = recover_bundle(np.arange(80), y, np.arange(68), settings=fast_gp)
    assert results['fprm'].error
    assert results['multiscale_equal'].error and results['multiscale_weighted'].error
    assert results['fprm_tau1'].error is None


def test_configuration_case_counts_and_reproduction_range():
    cfg = load_config(PROJECT / 'configs' / 'default.yaml')
    planned = cases(cfg, 'full')
    assert len(planned) == 960
    assert len(cases(cfg, 'smoke')) == 1
    assert {c['n'] for c in planned if c['experiment'] == 'reproduction'} == {416}
    assert {c['n'] for c in planned if c['experiment'] != 'reproduction'} == {410}
    assert sum(len(c['methods']) for c in planned) == 6360
    assert cases(cfg, 'smoke')[0]['target'] == 'NINO4'
    assert cases(cfg, 'smoke')[0]['methods'] == list(ALL_METHODS)


@pytest.mark.parametrize('suite', ['smoke', 'full'])
@pytest.mark.parametrize('changes,message', [
    ({'gp': {'delays': [1, 3, 211]}}, 'n=0'),
    ({'gp': {'delays': [1, 3, 212]}}, 'n=-2'),
    ({'gp': {'dimension': 72}}, 'n=-4'),
    ({'gp': {'delays': [1, 3, 210]}}, 'at least two observed labels'),
    ({'gp': {'folds': 206}}, 'CV folds'),
])
def test_unusable_embedding_or_cv_rejected_before_run_artifacts(tmp_path, monkeypatch, suite, changes, message):
    from enso_fprm import experiment
    cfg = load_config(PROJECT / 'configs/default.yaml')
    cfg['gp'].update(changes['gp'])
    cfg['rates'] = [0.5]
    artifact_root = tmp_path / 'unstarted_run'
    monkeypatch.setattr(experiment, 'PROC', artifact_root)
    def unexpected_prepare():
        pytest.fail('Unusable case reached data preparation')
    monkeypatch.setattr(experiment, 'prepare_assets', unexpected_prepare)
    with pytest.raises(ValueError, match=message) as error:
        cases(cfg, suite)
    assert 'dimension=' in str(error.value) and 'delays=' in str(error.value)
    with pytest.raises(ValueError, match=message):
        experiment.run(cfg, suite=suite)
    assert not artifact_root.exists()


@pytest.mark.parametrize('rate,folds,message', [
    (0.001, 3, 'one observed and one hidden'),
    (0.999, 3, 'at least two observed labels'),
    (0.995, 2, 'CV folds'),
])
def test_mask_and_cv_counts_checked_for_actual_full_cases(monkeypatch, rate, folds, message):
    from enso_fprm import experiment
    cfg = load_config(PROJECT / 'configs/default.yaml')
    cfg.update(rates=[rate], experiments=['main'])
    cfg['gp']['folds'] = folds
    monkeypatch.setattr(experiment, 'prepare_assets', lambda: pytest.fail('Invalid mask reached data preparation'))
    with pytest.raises(ValueError, match=message):
        experiment.run(cfg, suite='full')
    # Smoke uses its own fixed 50% rate, not the rates of the full suite.
    assert cases(cfg, 'smoke')[0]['rate'] == 0.5


def test_reproduction_ignores_unused_delays_rates_and_cv_folds():
    cfg = load_config(PROJECT / 'configs/default.yaml')
    cfg.update(experiments=['reproduction'], rates=[0.001])
    cfg['gp'].update(delays=[1, 3, 211], folds=500)
    planned = cases(cfg, 'full')
    assert len(planned) == 60
    assert all(c['n'] == 416 and c['methods'] == ['fprm'] and c['rate'] == 0.5 for c in planned)
    reference = np.sin(np.arange(422) * .13)
    observed = np.cos(np.arange(416) * .13)
    observed[::2] = np.nan
    settings = GPSettings(delays=(1, 3, 211), folds=500, optimizer=False, restarts=0)
    result = recover(reference, observed, np.arange(416), method='fprm', settings=settings)
    assert result.error is None and np.isfinite(result.recovered).all()


def test_feasible_cv_boundary_and_dimension_one_large_delay():
    cfg = load_config(PROJECT / 'configs/default.yaml')
    cfg.update(rates=[0.995], experiments=['main'])
    assert cases(cfg, 'full')[0]['n'] == 410  # Three observed labels support three folds.
    cfg['gp'].update(dimension=1, delays=[1, 3, 211])
    assert cases(cfg, 'smoke')[0]['n'] == 422  # A one-coordinate view has no delayed span.


def completed_fixture(tmp_path, methods=('mean', 'linear')):
    cfg = load_config(PROJECT / 'configs/default.yaml')
    case = cases(cfg, 'smoke')[0]
    case['methods'] = list(methods)
    frame = pd.DataFrame({v: np.sin(np.arange(422) * .13) for v in VARIABLES},
                         index=pd.period_range('1990-01', periods=422, freq='M'))
    return run_case(frame, case, cfg, GPSettings(optimizer=False, restarts=0), tmp_path, 'abc')


def test_resume_requires_intact_outputs(tmp_path):
    record = completed_fixture(tmp_path)
    assert cached_record(tmp_path, 'abc') == record
    assert cached_record(tmp_path, 'different') is None
    Path(record['predictions_file']).write_text('changed\n', encoding='utf-8')
    assert cached_record(tmp_path, 'abc') is None


@pytest.mark.parametrize('problem', ['missing_method', 'duplicate_method', 'extra_method',
                                     'missing_metric', 'missing_undefined', 'wrong_row_case',
                                     'invalid_metric', 'invalid_status'])
def test_resume_rejects_incomplete_method_records(tmp_path, capsys, problem):
    original = completed_fixture(tmp_path)
    corrupt = copy.deepcopy(original)
    if problem == 'missing_method':
        corrupt['rows'].pop()
    elif problem == 'duplicate_method':
        corrupt['rows'][1] = copy.deepcopy(corrupt['rows'][0])
    elif problem == 'extra_method':
        extra = copy.deepcopy(corrupt['rows'][0])
        extra['method'] = 'fprm'
        corrupt['rows'].append(extra)
    elif problem in ('missing_metric', 'missing_undefined'):
        corrupt['rows'][0].pop('nrmse' if problem == 'missing_metric' else 'undefined')
    elif problem == 'wrong_row_case':
        corrupt['rows'][0]['target'] = 'NINO12'
    elif problem == 'invalid_metric':
        corrupt['rows'][0]['rmse'] = 'invalid'
    else:
        corrupt['rows'][0]['status'] = 'success'
    broken = tmp_path / 'attempt_0002/completed.json'
    write_json(broken, corrupt)
    preserved = broken.read_bytes()
    assert cached_record(tmp_path, 'abc', original['case']) == original
    assert 'CACHE_SKIPPED=' in capsys.readouterr().out
    assert broken.read_bytes() == preserved
    # If the incomplete attempt is the only one, it must be recomputed.
    only = tmp_path / 'only_broken'
    write_json(only / 'attempt_0001/completed.json', corrupt)
    assert cached_record(only, 'abc', original['case']) is None


@pytest.mark.parametrize('field', ['rmse', 'mae', 'nrmse', 'rho', 'undefined',
                                 'n_hidden', 'n_observed', 'actual_rate', 'mask_sha256', 'total_seconds'])
def test_resume_rejects_values_inconsistent_with_saved_predictions(tmp_path, capsys, field):
    original = completed_fixture(tmp_path)
    changed = copy.deepcopy(original)
    row = changed['rows'][0]
    if field == 'undefined':
        row[field] = {}
    elif field == 'mask_sha256':
        row[field] = '0' * 64
    elif field in ('n_hidden', 'n_observed'):
        row[field] += 1
    else:
        row[field] = 0.0 if row[field] is None else row[field] * .5
    broken = tmp_path / 'attempt_0002/completed.json'
    write_json(broken, changed)
    preserved = broken.read_bytes()
    assert cached_record(tmp_path, 'abc', original['case']) == original
    assert 'CACHE_SKIPPED=' in capsys.readouterr().out
    only = tmp_path / 'only_changed'
    write_json(only / 'attempt_0001/completed.json', changed)
    assert cached_record(only, 'abc', original['case']) is None
    assert broken.read_bytes() == preserved
    # Direct aggregation cannot export a corrupted in-memory record either.
    export = tmp_path / 'exports'
    with pytest.raises(ValueError):
        aggregate([changed], export, load_config(PROJECT / 'configs/default.yaml'),
                  plots=False, planned=[original['case']])
    assert not export.exists()


@pytest.mark.parametrize('problem', ['empty', 'missing', 'duplicate', 'weight', 'oof_mse',
                                     'delay', 'case_id', 'method', 'missing_field'])
def test_resume_rejects_weights_inconsistent_with_saved_parameters(tmp_path, capsys, problem):
    original = completed_fixture(tmp_path, methods=('mean', 'multiscale_equal', 'multiscale_weighted'))
    assert cached_record(tmp_path, 'abc', original['case']) == original
    changed = copy.deepcopy(original)
    if problem == 'empty':
        changed['weights'] = []
    elif problem == 'missing':
        changed['weights'].pop()
    elif problem == 'duplicate':
        changed['weights'][-1] = copy.deepcopy(changed['weights'][0])
    elif problem == 'missing_field':
        changed['weights'][0].pop('weight')
    else:
        row = changed['weights'][-1]
        row[problem] = {'weight': 0.5, 'oof_mse': 1e10, 'delay': 99,
                        'case_id': 'wrong_case', 'method': 'mean'}[problem]
    write_json(tmp_path / 'attempt_0002/completed.json', changed)
    assert cached_record(tmp_path, 'abc', original['case']) == original
    assert 'CACHE_SKIPPED=' in capsys.readouterr().out
    only = tmp_path / 'only_changed'
    write_json(only / 'attempt_0001/completed.json', changed)
    assert cached_record(only, 'abc', original['case']) is None
    export = tmp_path / 'exports'
    with pytest.raises(ValueError):
        aggregate([changed], export, load_config(PROJECT / 'configs/default.yaml'),
                  plots=False, planned=[original['case']])
    assert not export.exists()


def test_resume_accepts_reordered_intact_weights(tmp_path):
    record = completed_fixture(tmp_path, methods=('multiscale_equal', 'multiscale_weighted'))
    record['weights'].reverse()
    write_json(tmp_path / 'attempt_0001/completed.json', record)
    assert cached_record(tmp_path, 'abc', record['case']) == record


def test_resume_rejects_record_for_another_planned_case(tmp_path, capsys):
    record = completed_fixture(tmp_path)
    expected = copy.deepcopy(record['case'])
    expected['seed'] = 1
    assert cached_record(tmp_path, 'abc', expected) is None
    assert 'planned case' in capsys.readouterr().out


def test_resume_rejects_missing_prediction_method_column(tmp_path, capsys):
    record = completed_fixture(tmp_path)
    path = Path(record['predictions_file'])
    table = pd.read_csv(path).drop(columns='linear')
    table.to_csv(path, index=False)
    record['files'][str(path)] = digest(path)
    write_json(tmp_path / 'attempt_0001/completed.json', record)
    assert cached_record(tmp_path, 'abc', record['case']) is None
    assert 'Prediction table' in capsys.readouterr().out


@pytest.mark.parametrize('problem', ['missing_method', 'duplicate_method', 'missing_field',
                                     'missing_case', 'wrong_case'])
def test_aggregation_refuses_incomplete_planned_results_before_writing(tmp_path, problem):
    record = completed_fixture(tmp_path)
    planned = [copy.deepcopy(record['case'])]
    damaged = copy.deepcopy(record)
    if problem == 'missing_method':
        damaged['rows'].pop()
    elif problem == 'duplicate_method':
        damaged['rows'][1] = copy.deepcopy(damaged['rows'][0])
    elif problem == 'missing_field':
        damaged['rows'][0].pop('undefined')
    elif problem == 'wrong_case':
        planned[0]['seed'] = 1
    export = tmp_path / 'exports'
    cfg = load_config(PROJECT / 'configs/default.yaml')
    with pytest.raises(ValueError):
        aggregate([] if problem == 'missing_case' else [damaged], export, cfg,
                  plots=False, planned=planned)
    assert not export.exists()


def test_aggregation_counts_complete_results_and_keeps_failures_explicit(tmp_path):
    record = completed_fixture(tmp_path)
    record['rows'][1].update(status='failed', error='controlled model failure',
                             rmse=None, mae=None, nrmse=None, rho=None)
    cfg = load_config(PROJECT / 'configs/default.yaml')
    checks = aggregate([record], tmp_path / 'exports', cfg, plots=False, planned=[record['case']])
    assert checks['n_expected_cases'] == checks['n_cases'] == 1
    assert checks['n_expected_method_results'] == checks['n_method_results'] == 2
    assert checks['n_failed'] == 1


@pytest.mark.parametrize('field,value', [
    ('bootstrap_iterations', 10000.0), ('bootstrap_iterations', True),
    ('bootstrap_iterations', 0), ('bootstrap_iterations', '10000'),
    ('statistics_seed', -1), ('statistics_seed', 0.0), ('statistics_seed', True),
    ('master_seed', -1), ('master_seed', True), ('plot_seed', -1), ('plot_seed', 0.0),
    ('seeds', [True]), ('seeds', [0.0]),
])
def test_invalid_integer_configuration_rejected_before_any_run_work(tmp_path, monkeypatch, field, value):
    from enso_fprm import experiment
    cfg = load_config(PROJECT / 'configs/default.yaml')
    cfg[field] = value
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match=field):
        load_config(path)
    def unexpected_prepare():
        pytest.fail('Invalid configuration reached data preparation')
    monkeypatch.setattr(experiment, 'prepare_assets', unexpected_prepare)
    with pytest.raises(ValueError, match=field):
        experiment.run(cfg)


@pytest.mark.parametrize('field,value', [('dimension', 3.0), ('delays', [1, 3, 6.0]),
                                       ('original_delay', True), ('folds', 3.0),
                                       ('restarts', 2.0), ('optimizer', 1), ('alphas', [float('nan')])])
def test_invalid_gp_parameter_types_rejected_at_config_load(tmp_path, field, value):
    cfg = load_config(PROJECT / 'configs/default.yaml')
    cfg['gp'][field] = value
    path = tmp_path / 'invalid_gp.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match=field):
        load_config(path)


@pytest.mark.parametrize('differences', [[], [0.1], [0.1, -0.2]])
@pytest.mark.parametrize('kwargs,field', [({'iterations': 10000.0}, 'bootstrap_iterations'),
                                         ({'iterations': True}, 'bootstrap_iterations'),
                                         ({'seed': -1}, 'statistics_seed')])
def test_statistics_api_rejects_invalid_parameters_even_without_bootstrap_sampling(differences, kwargs, field):
    with pytest.raises(ValueError, match=field):
        paired_bootstrap(differences, **kwargs)


@pytest.mark.parametrize('reference', VARIABLES)
def test_smoke_target_differs_from_reference(reference):
    cfg = load_config(PROJECT / 'configs/default.yaml')
    cfg['reference'] = reference
    cfg['alternate_reference'] = next(v for v in VARIABLES if v != reference)
    for case in cases(cfg, 'smoke') + cases(cfg, 'full'):
        assert case['reference'] != case['target']


def test_run_case_rejects_self_reference_before_writing(tmp_path):
    folder = tmp_path / 'self_reference'
    with pytest.raises(ValueError, match='must differ'):
        run_case(None, {'reference': 'NINO4', 'target': 'NINO4'}, None, None, folder, 'abc')
    assert not folder.exists()


@pytest.mark.parametrize('delays', [(1, 3), (2, 3, 9), (3,)])
def test_configured_delays_generate_executable_ablations(tmp_path, delays):
    cfg = load_config(PROJECT / 'configs/default.yaml')
    cfg['gp']['delays'] = list(delays)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    cfg = load_config(path)
    expected = {f'fprm_tau{d}' for d in delays if d != 3}
    for case in cases(cfg, 'smoke') + cases(cfg, 'full'):
        actual = {m for m in case['methods'] if m.startswith('fprm_tau')}
        assert actual == (set() if case['experiment'] == 'reproduction' else expected)
    settings = GPSettings(delays=delays, optimizer=False, restarts=0)
    reference = np.sin(np.arange(90) * .13)
    n = 90 - 2 * max(delays)
    observed = np.cos(np.arange(n) * .13)
    observed[::2] = np.nan
    default_results = recover_bundle(reference, observed, np.arange(n), settings=settings)
    smoke_results = recover_bundle(reference, observed, np.arange(n), settings=settings,
                                   methods=cases(cfg, 'smoke')[0]['methods'])
    assert set(default_results) == set(smoke_results)
    for method, result in default_results.items():
        assert result.error is None
        np.testing.assert_array_equal(result.recovered, smoke_results[method].recovered)
    assert default_results['multiscale_weighted'].metadata['delays'] == list(delays)


@pytest.mark.parametrize('method', ['mean', 'linear', 'gpr_raw'])
def test_standalone_baselines_accept_full_reference_range(method, fast_gp):
    reference = np.sin(np.arange(80) * .13)
    observed = np.cos(np.arange(80) * .13)
    observed[::2] = np.nan
    result = recover(reference, observed, np.arange(80), method=method, settings=fast_gp)
    assert result.error is None
    assert np.isfinite(result.recovered).all()
    np.testing.assert_array_equal(result.recovered[1::2], observed[1::2])


def test_baselines_share_original_fprm_range(fast_gp):
    reference = np.sin(np.arange(80) * .13)
    observed = np.cos(np.arange(74) * .13)
    observed[::2] = np.nan
    results = recover_bundle(reference, observed, np.arange(74), settings=fast_gp,
                             methods=['mean', 'linear', 'gpr_raw', 'fprm'])
    assert all(r.error is None for r in results.values())
    with pytest.raises(ValueError, match='outside valid embedding range'):
        recover_bundle(reference, observed, np.arange(74), settings=fast_gp,
                       methods=['mean', 'multiscale_equal'])


@pytest.mark.parametrize('delay', [1, 2, 6])
def test_explicit_single_scale_uses_its_own_valid_range(delay, fast_gp):
    reference = np.sin(np.arange(80) * .13)
    n = 80 - 2 * delay
    observed = np.cos(np.arange(n) * .13)
    observed[::2] = np.nan
    result = recover(reference, observed, np.arange(n), method=f'fprm_tau{delay}', settings=fast_gp)
    assert result.error is None
    assert result.metadata['delay'] == delay
    assert len(result.recovered) == n


def test_extra_ablation_is_excluded_from_fusion_and_its_failures(monkeypatch):
    from enso_fprm import models
    settings = GPSettings(delays=(1, 3), optimizer=False, restarts=0)
    reference = np.arange(80, dtype=float)
    observed = np.sin(np.arange(68) * .13)
    observed[::2] = np.nan
    idx = np.arange(68)
    methods = ['fprm_tau6', 'multiscale_weighted']
    results = recover_bundle(reference, observed, idx, settings=settings, methods=methods)
    assert results['fprm_tau6'].error is None
    weighted = results['multiscale_weighted']
    assert weighted.error is None and weighted.metadata['delays'] == [1, 3]
    assert len(weighted.metadata['weights']) == 2
    original = models.fit_predict
    def fail_extra(x, y, pred, settings, seed):
        if np.allclose(x[:, 1] - x[:, 0], 6):
            raise ModelFailure('controlled extra ablation failure')
        return original(x, y, pred, settings, seed)
    monkeypatch.setattr(models, 'fit_predict', fail_extra)
    after = recover_bundle(reference, observed, idx, settings=settings, methods=methods)
    assert after['fprm_tau6'].error
    assert after['multiscale_weighted'].error is None
    np.testing.assert_array_equal(after['multiscale_weighted'].recovered, weighted.recovered)


@pytest.mark.parametrize('method', ['fprm_tau0', 'fprm_tau03', 'fprm_tau-1', 'fprm_taufoo'])
def test_invalid_ablation_names_are_rejected(method, sample, fast_gp):
    reference, _, observed, idx = sample
    with pytest.raises(ValueError, match='Unknown'):
        recover(reference, observed, idx, method=method, settings=fast_gp)


@pytest.mark.parametrize('content', ['{"signature":', '[]', '{"signature":"abc"}'])
def test_resume_skips_corrupt_latest_record_without_changing_it(tmp_path, capsys, content):
    record = completed_fixture(tmp_path)
    broken = tmp_path / 'attempt_0002' / 'completed.json'
    broken.parent.mkdir()
    broken.write_text(content, encoding='utf-8')
    assert cached_record(tmp_path, 'abc') == record
    assert str(broken) in capsys.readouterr().out
    assert broken.read_text(encoding='utf-8') == content


def test_resume_skips_record_with_empty_output_checksums(tmp_path, capsys):
    record = completed_fixture(tmp_path)
    record['files'] = {}
    write_json(tmp_path / 'attempt_0001' / 'completed.json', record)
    assert cached_record(tmp_path, 'abc') is None
    assert 'CACHE_SKIPPED=' in capsys.readouterr().out


def test_resume_creates_new_attempt_after_interrupted_record(tmp_path, fast_gp):
    broken = tmp_path / 'attempt_0001' / 'completed.json'
    broken.parent.mkdir()
    broken.write_text('{"signature":', encoding='utf-8')
    assert cached_record(tmp_path, 'abc') is None
    frame = pd.DataFrame({v: np.sin(np.arange(422) * .13) for v in VARIABLES},
                         index=pd.period_range('1990-01', periods=422, freq='M'))
    cfg = load_config(PROJECT / 'configs/default.yaml')
    case = cases(cfg, 'smoke')[0]
    case['methods'] = ['mean']
    record = run_case(frame, case, cfg, fast_gp, tmp_path, 'abc')
    assert Path(record['predictions_file']).parent.name == 'attempt_0002'
    assert cached_record(tmp_path, 'abc') == record
    assert broken.read_text(encoding='utf-8') == '{"signature":'


def test_bootstrap_pairs_seeds_and_undefined_single_seed():
    a = paired_bootstrap([-0.4, 0.1, -0.1], iterations=1000, seed=8)
    assert a == paired_bootstrap([-0.4, 0.1, -0.1], iterations=1000, seed=8)
    assert a['delta_mean'] == pytest.approx(-0.4/3)
    assert a['win_rate'] == pytest.approx(2/3)
    assert a['ci_lower'] <= a['delta_mean'] <= a['ci_upper']
    assert paired_bootstrap([0.1])['ci_lower'] is None


def test_overall_does_not_treat_targets_as_independent_seeds():
    rows = []
    for target, value in [('NINO4', 1), ('NINO3', 2), ('NINO12', 3)]:
        rows.append(dict(experiment='main', reference='NINO34', mask_kind='random', rate=0.5,
                         n=410, method='fprm', seed=0, expected_targets=3, target=target, status='ok', nrmse=value))
    result = seed_overall(pd.DataFrame(rows))
    assert len(result) == 1 and result.nrmse.iloc[0] == 2
    rows[0]['status'] = 'failed'
    assert pd.isna(seed_overall(pd.DataFrame(rows)).nrmse.iloc[0])
