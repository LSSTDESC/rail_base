"""Add a bias to the redshift using bias values from the literature.
   This takes true redshifts and produces redshifts with noise levels intended to match those in COSMOS2020."""

from typing import Any

import numpy as np
from ceci.config import StageParameter as Param

from rail.creation.selector import Selector

class COSMOSSelector(Selector):
    """Add a column of random numbers to a dataframe"""

    name = "COSMOSSelector"
    entrypoint_function = "__call__"  # the user-facing science function for this class
    interactive_function = "COSMOSSelector"
    config_options = Selector.config_options.copy()
    config_options.update(
        col_name=Param(
            str, "photoz_COSMOS2025", msg="Name of the column with mock photometric redshifts"
        ),
        col_name_mag_i=Param(
            str, "mag_i", msg="Name of the mag_i column"
        ),
        col_name_z=Param(
            str, "z", msg="Name of the redshift column"
        ),
    )

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """
        Constructor
        Does standard Selector initialization
        """
        Selector.__init__(self, args, **kwargs)

    def _initNoiseModel(self) -> None:  # pragma: no cover
        self._rng = np.random.default_rng(self.config.seed)

    def _addNoise(self) -> None:  # pragma: no cover
        self._addNoiseCOSMOS()

    def _select(self) -> None:  # pragma: no cover
        # for this selector, we currently don't actually select any rows
        data = self.get_data("input")
        selection_mask = np.ones(len(data), dtype=bool)

        # for the COSMOS selector, emulate photo-z's
        self._initNoiseModel()
        self._addNoise()
        return selection_mask
    
    def COSMOSSelector(self, sample: Any, seed: int | None = None, **kwargs: Any):
        return self.__call__(sample, seed=seed, **kwargs)

    def _sample_parametric_bias_model(
        self,
        data_i: np.ndarray,
        data_z: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Sample bias using a parametric model with three components:
           core (Yin+25), tails (Gaussian), and outliers (flat)."""
        z_bias_samples = np.zeros_like(data_z, dtype=float)
        if not np.any(target_mask):
            return z_bias_samples

        # Bin edges and lookup tables for the core component (Yin+25) encoded directly here
        mag_i_bin_edges = np.array([15.5, 22, 23, 24, 29])
        z_bin_edges = np.array([0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        bias_median_lookup_table = np.array(
            [
                [0.000, 0.002, -0.002, 0.001, 0.001, 0.001, 0.001, 0.001],
                [0.000, -0.000, -0.002, -0.004, 0.010, 0.010, 0.010, 0.010],
                [0.003, -0.000, -0.001, -0.006, -0.002, 0.024, 0.007, 0.000],
                [0.008, -0.005, 0.007, -0.015, -0.019, 0.017, 0.011, 0.000],
            ]
        )
        bias_std_lookup_table = np.array(
            [
                [0.010, 0.020, 0.026, 0.038, 0.038, 0.038, 0.038, 0.038],
                [0.011, 0.019, 0.025, 0.036, 0.062, 0.062, 0.062, 0.062],
                [0.011, 0.022, 0.027, 0.044, 0.063, 0.115, 0.093, 0.074],
                [0.013, 0.023, 0.025, 0.051, 0.069, 0.120, 0.103, 0.069],
            ]
        )
        # Additional encoded tails and outliers in bins of mag_i
        f_tail_by_mag_i = np.array([0.03, 0.06, 0.10, 0.15], dtype=float)
        f_outlier_by_mag_i = np.array([0.001, 0.003, 0.008, 0.020], dtype=float)
        tail_std_by_mag_i = np.array([0.21, 0.31, 0.43, 0.60], dtype=float)
        outlier_high_by_mag_i = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)


        n_target = int(np.sum(target_mask))
        target_i = data_i[target_mask]
        target_z = data_z[target_mask]

        # determine mag_i and z bin for each target galaxy
        # if the target falls outside the bin edges, assign it to the nearest bin
        target_mag_i_bin = np.digitize(target_i, mag_i_bin_edges) - 1
        target_z_bin = np.digitize(target_z, z_bin_edges) - 1
        target_mag_i_bin = np.clip(target_mag_i_bin, 0, bias_median_lookup_table.shape[0] - 1)
        target_z_bin = np.clip(target_z_bin, 0, bias_median_lookup_table.shape[1] - 1)


        # set component parameters for each target
        target_mean_bias_component1 = bias_median_lookup_table[target_mag_i_bin, target_z_bin]
        target_std_bias_component1 = bias_std_lookup_table[target_mag_i_bin, target_z_bin]
        target_outlier_prob = f_outlier_by_mag_i[target_mag_i_bin]
        target_tail_prob = f_tail_by_mag_i[target_mag_i_bin]

        # Monte Carlo sampling to determine which component each galaxy belongs to.
        u = self._rng.random(n_target)
        is_outlier = u < target_outlier_prob
        is_tail = (~is_outlier) & (u < (target_outlier_prob + target_tail_prob))
        is_core = ~(is_tail | is_outlier)

        # generate redshift bias for each galaxy based on its assigned component
        target_bias = np.zeros(n_target, dtype=float)
        target_bias[is_core] = self._rng.normal(
            loc=target_mean_bias_component1[is_core],
            scale=target_std_bias_component1[is_core],
        )
        if np.any(is_tail):
            target_bias[is_tail] = self._rng.normal(
                loc=0.0,
                scale=tail_std_by_mag_i[target_mag_i_bin[is_tail]],
                size=np.sum(is_tail),
            )
        if np.any(is_outlier):
            target_bias[is_outlier] = self._rng.uniform(
                low=0.0,
                high=outlier_high_by_mag_i[target_mag_i_bin[is_outlier]],
                size=np.sum(is_outlier),
            )
        z_bias_samples[target_mask] = target_bias
        return z_bias_samples

    def _addNoiseCOSMOS(self) -> None:  # pragma: no cover
        data = self.get_data("input")
        data_i = np.asarray(data[self.config.col_name_mag_i], dtype=float)
        data_z = np.asarray(data[self.config.col_name_z], dtype=float)
        valid_target_mask = np.isfinite(data_i) & np.isfinite(data_z) & (data_z > 0)
        z_bias_samples = self._sample_parametric_bias_model(
            data_i=data_i,
            data_z=data_z,
            target_mask=valid_target_mask,
        )

        # add the bias values to the true redshifts and clip at zero
        z_noisified = data_z + z_bias_samples
        z_noisified = np.clip(z_noisified, 0, None) # ensure no negative redshifts

        data[self.config.col_name] = z_noisified
        return