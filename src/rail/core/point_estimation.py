import numpy as np
from numpy.typing import NDArray


class PointEstimationMixin:
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
        if "calculated_point_estimates" in self.config:
            calculated_point_estimates = self.config["calculated_point_estimates"]

        if "mode" in calculated_point_estimates:
            mode_value = self._calculate_mode_point_estimate(qp_dist, grid)
            ancil_dict.update(mode=mode_value)

        if "mean" in calculated_point_estimates:
            mean_value = self._calculate_mean_point_estimate(qp_dist)
            ancil_dict.update(mean=mean_value)

        if "median" in calculated_point_estimates:
            median_value = self._calculate_median_point_estimate(qp_dist)
            ancil_dict.update(median=median_value)

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
