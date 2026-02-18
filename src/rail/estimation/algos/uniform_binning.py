"""
A classifier that uses pz point estimate to assign
tomographic bins with uniform binning.
"""

import numpy as np
import qp
from ceci.config import StageParameter as Param

from rail.core.data import Hdf5Handle
from rail.core.common_params import SharedParams
from rail.estimation.classifier import PZClassifier


class UniformBinningClassifier(PZClassifier):
    """Classifier that simply assigns tomographic bins based on a point estimate
    according to SRD.
    """

    name = "UniformBinningClassifier"
    entrypoint_function = "classify"  # the user-facing science function for this class
    interactive_function = "uniform_binning_classifier"
    config_options = PZClassifier.config_options.copy()
    config_options.update(
        object_id_col=SharedParams.copy_param("object_id_col"),
        point_estimate_key=Param(str, "zmode", msg="Which point estimate to use"),
        zbin_edges=Param(
            list,
            [],
            msg="The tomographic redshift bin edges."
            "If this is given (contains two or more entries), all settings below will be ignored.",
        ),
        zmin=SharedParams.copy_param("zmin"),  # minimum redshift of the sample
        zmax=SharedParams.copy_param("zmax"),  # Maximum redshift of the sample
        n_tom_bins=Param(int, 5, msg="Number of tomographic bins"),
        no_assign=Param(int, -99, msg="Value for no assignment flag"),
    )

    # update the default object id column name in the input data to be empty, which uses the row index as the ID
    config_options["object_id_col"].set_default("")

    outputs = [("output", Hdf5Handle)]

    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, first: bool
    ) -> None:
        """Process a chunk of data for uniform binning classification.

        Parameters
        ----------
        start
            The starting index of the chunk

        end
            The ending index of the chunk

        data
            The data chunk to be processed

        first
            True if this is the first chunk, False otherwise.
        """
        try:
            zb = data.ancil[self.config.point_estimate_key]
        except KeyError as missing_key:
            raise KeyError(
                f"{self.config.point_estimate_key} is not contained "
                "in the data ancil, you will need to compute it explicitly."
            ) from missing_key

        try:
            _npdf = data.npdf
        except KeyError as missing_key:
            raise KeyError(
                f"npdf is not a supported attribute of {type(data)}. "
                "Are you sure you don't mean to be using a qp ensemble?"
            ) from missing_key

        # binning options
        if len(self.config.zbin_edges) >= 2:
            # this overwrites all other key words
            # linear binning defined by zmin, zmax, and n_tom_bins
            bin_index = np.digitize(zb, self.config.zbin_edges)
            # assign -99 to objects not in any bin:
            bin_index[bin_index == 0] = self.config.no_assign
            bin_index[bin_index == len(self.config.zbin_edges)] = self.config.no_assign
        else:
            # linear binning defined by zmin, zmax, and n_tom_bins
            bin_index = np.digitize(
                zb,
                np.linspace(
                    self.config.zmin, self.config.zmax, self.config.n_tom_bins + 1
                ),
            )
            # assign -99 to objects not in any bin:
            bin_index[bin_index == 0] = self.config.no_assign
            bin_index[bin_index == (self.config.n_tom_bins + 1)] = self.config.no_assign

        if self.config.object_id_col != "":
            # below is commented out and replaced by a redundant line
            # because the data doesn't have ID yet
            # obj_id = data[self.config.object_id_col]
            obj_id = np.arange(data.npdf)
        elif self.config.object_id_col == "":
            # ID set to row index
            obj_id = np.arange(data.npdf)
            self.config.object_id_col = "row_index"

        class_id = {
            self.config.object_id_col: obj_id,  # pylint: disable=possibly-used-before-assignment
            "class_id": bin_index,
        }
        self._do_chunk_output(class_id, start, end, first)
