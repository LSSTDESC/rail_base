import numpy as np
from numpy.typing import NDArray

from rail.core.common_params import SHARED_PARAMS


class PointEstimationMixin:

    config_options = dict(
        calculated_point_estimates=SHARED_PARAMS,
        recompute_point_estimates=SHARED_PARAMS,
    )    
    
    def calculate_point_estimates(self, qp_dist, grid=None):
        """This function drives the calculation of point estimates for qp.Ensembles.
        It is defined here, and called from the `_process_chunk` method in the
        `CatEstimator` child classes.

        Parameters
        ----------
        qp_dist : qp.Ensemble
            The qp Ensemble instance that contains posterior estimates.
        grid : array-like, optional
            The grid on which to evaluate the point estimate. Note that not all
            point estimates require a grid to be provided, by default None.

        Returns
        -------
        qp.Ensemble
            The original `qp.Ensemble` with new ancillary point estimate data included.
            The `Ensemble.ancil` keys are ['mean', 'mode', 'median'].

        Notes
        -----
        If there are particularly efficient ways to calculate point estimates for
        a given `CatEstimator` subclass, the subclass can implement any of the
        `_calculate_<foo>_point_estimate` for any of the point estimate calculator
        methods, i.e.:

            - `_calculate_mode_point_estimate`
            - `_calculate_mean_point_estimate`
            - `_calculate_median_point_estimate`
        """

        ancil_dict = dict()
        calculated_point_estimates = []

        existing_ancil = qp_dist.ancil
        if existing_ancil and not self.config.recompute_point_estimates:
            skip_zmode = 'zmode' in existing_ancil
            skip_zmean = 'zmean' in existing_ancil
            skip_zmedian = 'zmedian' in existing_ancil
        else:
            skip_zmode = False
            skip_zmean = False
            skip_zmedian = False            
        
        if "calculated_point_estimates" in self.config:
            calculated_point_estimates = self.config["calculated_point_estimates"]

        if "zmode" in calculated_point_estimates and not skip_zmode:
            mode_value = self._calculate_mode_point_estimate(qp_dist, grid)
            ancil_dict.update(zmode=mode_value)

        if "zmean" in calculated_point_estimates and not skip_zmean:
            mean_value = self._calculate_mean_point_estimate(qp_dist)
            ancil_dict.update(zmean=mean_value)

        if "zmedian" in calculated_point_estimates and not skip_zmedian:
            median_value = self._calculate_median_point_estimate(qp_dist)
            ancil_dict.update(zmedian=median_value)

        if calculated_point_estimates:
            if qp_dist.ancil is None:
                qp_dist.set_ancil(ancil_dict)
            else:
                qp_dist.add_to_ancil(ancil_dict)

        return qp_dist

    def _calculate_mode_point_estimate(self, qp_dist, grid=None) -> NDArray:
        """Calculates and returns the mode values for a set of posterior estimates
        in a qp.Ensemble instance.

        Parameters
        ----------
        qp_dist : qp.Ensemble
            The qp Ensemble instance that contains posterior estimates.
        grid : array-like, optional
            The grid on which to evaluate the `mode` point estimate, if a grid is
            not provided, a default will be created at run time using `zmin`, `zmax`,
            and `nzbins`, by default None

        Returns
        -------
        NDArray
            The mode value for each posterior in the qp.Ensemble

        Raises
        ------
        KeyError
            If `grid` is not provided, one will be created using stage_config
            `zmin`, `zmax`, and `nzbins` keys. If any of those keys are missing,
            we'll raise a KeyError.
        """
        if grid is None:
            for key in ["zmin", "zmax", "nzbins"]:
                if key not in self.config:
                    raise KeyError(
                        f"Expected `{key}` to be defined in stage "
                        "configuration dictionary in order to caluclate mode."
                    )

            grid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        return qp_dist.mode(grid=grid)

    def _calculate_mean_point_estimate(self, qp_dist) -> NDArray:
        """Calculates and returns the mean values for a set of posterior estimates
        in a qp.Ensemble instance.

        Parameters
        ----------
        qp_dist : qp.Ensemble
            The qp Ensemble instance that contains posterior estimates.

        Returns
        -------
        NDArray
            The mean value for each posterior in the qp.Ensemble
        """
        return qp_dist.mean()

    def _calculate_median_point_estimate(self, qp_dist) -> NDArray:
        """Calculates and returns the median values for a set of posterior estimates
        in a qp.Ensemble instance.

        Parameters
        ----------
        qp_dist : qp.Ensemble
            The qp Ensemble instance that contains posterior estimates.

        Returns
        -------
        NDArray
            The median value for each posterior in the qp.Ensemble
        """
        return qp_dist.median()
