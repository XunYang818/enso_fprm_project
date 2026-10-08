from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from enso_fprm.data import load_enso, prepare_assets
from enso_fprm.embedding import delay_embedding
from enso_fprm.experiment import cached_record, cases, load_config
from enso_fprm.io import PROJECT, PROC, canonical_hash, digest, write_json
from enso_fprm.masks import make_mask
from enso_fprm.metrics import evaluate
from enso_fprm.models import GPSettings, ModelFailure, fit_predict, recover_bundle, standardize
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


def test_resume_requires_intact_outputs(tmp_path):
    f = tmp_path / 'attempt_0001' / 'predictions.csv'
    f.parent.mkdir()
    f.write_text('a\n1\n', encoding='utf-8')
    record = {'signature': 'abc', 'rows': [{'status': 'ok'}], 'files': {str(f): digest(f)}}
    write_json(f.parent / 'completed.json', record)
    assert cached_record(tmp_path, 'abc') == record
    assert cached_record(tmp_path, 'different') is None
    f.write_text('a\n2\n', encoding='utf-8')
    assert cached_record(tmp_path, 'abc') is None


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
