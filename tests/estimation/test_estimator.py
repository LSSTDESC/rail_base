import pytest
import numpy as np

import qp
from rail.estimation.estimator import CatEstimator


def test_custom_point_estimate():
    """This test checks to make sure that the inheritance mechanism is working
    for child classes of `CatEstimator`.
    """

    MEANING_OF_LIFE = 42.0
    class TestEstimator(CatEstimator):
        name="TestEstimator"

        def __init__(self, args, comm=None):
            CatEstimator.__init__(self, args, comm=comm)

        def _calculate_mode_point_estimate(self, qp_dist=None, grid=None):
            return np.ones(100) * MEANING_OF_LIFE

    config_dict = {'calculated_point_estimates': ['mode']}

    test_estimator = TestEstimator.make_stage(name='test', **config_dict)

    locs = 2* (np.random.uniform(size=(100,1))-0.5)
    scales = 1 + 0.2*(np.random.uniform(size=(100,1))-0.5)
    test_ensemble = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))

    result = test_estimator._calculate_point_estimates(test_ensemble, None)

    assert np.all(result.ancil['mode'] == MEANING_OF_LIFE)

def test_basic_point_estimate():
    """This test checks to make sure that all the basic point estimates are
    executed when requested in the configuration dictionary.
    """

    config_dict = {'calculated_point_estimates': ['mean', 'median', 'mode'],
                   'zmin': 0.0,
                   'zmax': 3.0,
                   'nzbins': 301}

    test_estimator = CatEstimator.make_stage(name='test', **config_dict)

    locs = 2* (np.random.uniform(size=(100,1))-0.5)
    scales = 1 + 0.2*(np.random.uniform(size=(100,1))-0.5)
    test_ensemble = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))
    result = test_estimator._calculate_point_estimates(test_ensemble, None)

    # note: we're not interested in testing the values of point estimates,
    # just that they were added to the ancillary data.
    assert 'mode' in result.ancil
    assert 'median' in result.ancil
    assert 'mean' in result.ancil

def test_mode_no_grid():
    """This exercises the KeyError logic in `_calculate_mode_point_estimate`.
    """
    config_dict = {'zmin':0.0, 'nzbins':100, 'calculated_point_estimates': ['mode']}

    test_estimator = CatEstimator.make_stage(name='test', **config_dict)

    with pytest.raises(KeyError) as excinfo:
        _ = test_estimator._calculate_point_estimates(None, None)

    assert "to be defined in stage configuration" in str(excinfo.value)

def test_mode_no_point_estimates():
    """This exercises the KeyError logic in `_calculate_mode_point_estimate`.
    """
    config_dict = {'zmin':0.0, 'nzbins':100}

    test_estimator = CatEstimator.make_stage(name='test', **config_dict)

    locs = 2* (np.random.uniform(size=(100,1))-0.5)
    scales = 1 + 0.2*(np.random.uniform(size=(100,1))-0.5)
    test_ensemble = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))

    output_ensemble = test_estimator._calculate_point_estimates(test_ensemble, None)

    assert output_ensemble.ancil is None