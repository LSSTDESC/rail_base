"""
Abstract base classes defining Estimators of individual galaxy redshift uncertainties.
"""

import gc
from typing import Any, Optional

import numpy as np
import qp

from rail.core.common_params import SHARED_PARAMS
from rail.core.data import (DataHandle, ModelHandle, ModelLike, QPHandle,
                            TableHandle, TableLike)
from rail.core.point_estimation import PointEstimationMixin
from rail.core.stage import RailStage

# for backwards compatibility, to avoid break stuff that imports it from here
from .informer import CatInformer  # pylint: disable=unused-import


class CatEstimator(RailStage, PointEstimationMixin):
    """The base class for making photo-z posterior estimates from catalog-like inputs
    (i.e., tables with fluxes in photometric bands among the set of columns)

    Estimators use a generic "model", the details of which depends on the sub-class.

    Estimators take as "input" tabular data, apply the photo-z estimation and
    provide as "output" a ``QPEnsemble``, with per-object p(z).

    """

    name = "CatEstimator"
    config_options = RailStage.config_options.copy()
    config_options.update(
        chunk_size=SHARED_PARAMS,
        hdf5_groupname=SHARED_PARAMS,
        zmin=SHARED_PARAMS,
        zmax=SHARED_PARAMS,
        nzbins=SHARED_PARAMS,
        id_col=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
    )
    config_options.update(
        **PointEstimationMixin.config_options.copy(),
    )
    inputs = [("model", ModelHandle), ("input", TableHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Estimator"""
        super().__init__(args, **kwargs)
        self._output_handle: QPHandle | None = None
        self.model = None

    def estimate(self, input_data: TableLike) -> DataHandle:
        """The main interface method for the photo-z estimation

        This will attach the input data (defined in ``inputs`` as "input") to this
        ``Estimator`` (for introspection and provenance tracking). Then call the
        ``run()``, ``validate()``, and ``finalize()`` methods.

        The run method will call ``_process_chunk()``, which needs to be implemented
        in the subclass, to process input data in batches. See ``RandomGaussEstimator``
        for a simple example.

        Finally, this will return a ``QPHandle`` for access to that output data.

        Parameters
        ----------
        input_data
            A dictionary of all input data

        Returns
        -------
        DataHandle
            Handle providing access to QP ensemble with output data
        """
        self.set_data("input", input_data)
        self.validate()
        self.run()
        self.finalize()
        results = self.get_handle("output")
        results.read(force=True)
        return results

    def run(self) -> None:
        self.open_model(**self.config)

        iterator = self.input_iterator("input")
        first = True
        self._initialize_run()
        self._output_handle = None
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s:,} - {e:,}")
            self._process_chunk(s, e, test_data, first)
            first = False
            # Running garbage collection manually seems to be needed
            # to avoid memory growth for some estimators
            gc.collect()
        self._finalize_run()

    def _initialize_run(self) -> None:
        self._output_handle = None

    def _finalize_run(self) -> None:
        assert self._output_handle is not None
        self._output_handle.finalize_write()

    def _process_chunk(
        self, start: int, end: int, data: TableLike, first: bool
    ) -> None:
        raise NotImplementedError(
            f"{self.name}._process_chunk is not implemented"
        )  # pragma: no cover

    def _do_chunk_output(
        self,
        qp_dstn: qp.Ensemble,
        start: int,
        end: int,
        first: bool,
        data: Optional[TableLike] = None,
    ) -> None:
        qp_dstn = self.calculate_point_estimates(qp_dstn)

        # if there is no ancil set by the calculate_point_estimate, initiate one
        if data is not None:
            if qp_dstn.ancil is None:  # pragma: no cover
                ancil_dict: dict[str, np.ndarray] = dict()
                qp_dstn.set_ancil(ancil_dict)
            assert qp_dstn.ancil is not None
            # if there is ID column in the input dataset, attach it to the ancil
            if self.config.id_col in data.keys():  # pragma: no cover
                qp_dstn.ancil.update(id=data[self.config.id_col])
            # if there is redshift column in the input dataset, attach it to the ancil
            if self.config.redshift_col in data.keys():  # pragma: no cover
                qp_dstn.ancil.update(redshift=data[self.config.redshift_col])

        if first:
            the_handle = self.add_handle("output", data=qp_dstn)
            assert isinstance(the_handle, QPHandle)
            self._output_handle = the_handle
            if self.config.output_mode != "return":
                self._output_handle.initialize_write(
                    self._input_length, communicator=self.comm
                )
        assert self._output_handle is not None
        self._output_handle.set_data(qp_dstn, partial=True)
        if self.config.output_mode != "return":
            self._output_handle.write_chunk(start, end)
        return qp_dstn


class PzEstimator(RailStage, PointEstimationMixin):
    """The base class for making photo-z posterior estimates from other pz inputs

    Estimators use a generic "model", the details of which depends on the sub-class.

    Estimators take as "input" a QPEnsemble, with other estimates and
    provide as "output" a ``QPEnsemble``, with per-object p(z).

    """

    name = "PzEstimator"
    config_options = RailStage.config_options.copy()
    config_options.update(
        chunk_size=SHARED_PARAMS,
        hdf5_groupname=SHARED_PARAMS,
    )
    config_options.update(
        **PointEstimationMixin.config_options.copy(),
    )
    inputs = [("model", ModelHandle), ("input", QPHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Estimator"""
        super().__init__(args, **kwargs)
        self._output_handle: QPHandle | None = None
        self.model = None

    def estimate(self, input_data: QPHandle) -> DataHandle:
        """The main interface method for the photo-z estimation

        This will attach the input data (defined in ``inputs`` as "input") to this
        ``Estimator`` (for introspection and provenance tracking). Then call the
        ``run()``, ``validate()``, and ``finalize()`` methods.

        The run method will call ``_process_chunk()``, which needs to be implemented
        in the subclass, to process input data in batches. See ``RandomGaussEstimator``
        for a simple example.

        Finally, this will return a ``QPHandle`` for access to that output data.

        Parameters
        ----------
        input_data
            A dictionary of all input data

        Returns
        -------
        DataHandle
            Handle providing access to QP ensemble with output data
        """
        self.set_data("input", input_data)
        self.validate()
        self.run()
        self.finalize()
        results = self.get_handle("output")
        results.read(force=True)
        return results

    def run(self) -> None:
        self.open_model(**self.config)

        iterator = self.input_iterator("input")
        first = True
        self._initialize_run()
        self._output_handle = None
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s:,} - {e:,}")
            self._process_chunk(s, e, test_data, first)
            first = False
            # Running garbage collection manually seems to be needed
            # to avoid memory growth for some estimators
            gc.collect()
        self._finalize_run()

    def _initialize_run(self) -> None:
        self._output_handle = None

    def _finalize_run(self) -> None:
        assert self._output_handle is not None
        self._output_handle.finalize_write()

    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, first: bool
    ) -> None:
        raise NotImplementedError(
            f"{self.name}._process_chunk is not implemented"
        )  # pragma: no cover

    def _do_chunk_output(
        self, qp_dstn: qp.Ensemble, start: int, end: int, first: bool
    ) -> None:
        qp_dstn = self.calculate_point_estimates(qp_dstn)
        if first:
            the_handle = self.add_handle("output", data=qp_dstn)
            assert isinstance(the_handle, QPHandle)
            self._output_handle = the_handle
            if self.config.output_mode != "return":
                self._output_handle.initialize_write(
                    self._input_length, communicator=self.comm
                )
        assert self._output_handle is not None
        self._output_handle.set_data(qp_dstn, partial=True)
        if self.config.output_mode != "return":
            self._output_handle.write_chunk(start, end)
        return qp_dstn
