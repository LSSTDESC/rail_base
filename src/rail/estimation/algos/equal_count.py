"""
A classifier that uses pz point estimate to assign
tomographic bins with uniform binning.
"""

import numpy as np
from ceci.config import StageParameter as Param

from rail.core.data import Hdf5Handle
from rail.estimation.classifier import PZClassifier
from rail.core.common_params import SharedParams


class EqualCountClassifier(PZClassifier):
    """Classifier that simply assign tomographic
    bins based on point estimate according to SRD"""

    name = "EqualCountClassifier"
    entrypoint_function = "classify"  # the user-facing science function for this class
    interactive_function = "equal_count_classifier"
    config_options = PZClassifier.config_options.copy()
    config_options.update(
        object_id_col=SharedParams.copy_param("object_id_col"),
        point_estimate_key=Param(str, "zmode", msg="Which point estimate to use"),
        zmin=SharedParams.copy_param("zmin"),  # minimum redshift of the sample
        zmax=SharedParams.copy_param("zmax"),  # Maximum redshift of the sample
        n_tom_bins=Param(int, 5, msg="Number of tomographic bins"),
        no_assign=Param(int, -99, msg="Value for no assignment flag"),
    )

    # update the default object id column name in the input data to be empty, which uses the row index as the ID
    config_options["object_id_col"].set_default("")
    outputs = [("output", Hdf5Handle)]

    def run(self) -> None:
        test_data = self.get_data("input")
        npdf = test_data.npdf

        try:
            zb = np.squeeze(test_data.ancil[self.config.point_estimate_key])
        except KeyError as msg:
            raise KeyError(
                f"{self.config.point_estimate_key} is not contained in the data ancil, "
                "you will need to compute it explicitly."
            ) from msg

        # tomographic bins with equal number density
        sortind = np.argsort(zb)
        cum = np.arange(1, (len(zb) + 1))
        bin_index = np.zeros(len(zb))
        for ii in range(self.config.n_tom_bins):
            perc1 = ii / self.config.n_tom_bins
            perc2 = (ii + 1) / self.config.n_tom_bins
            ind = (cum / cum[-1] > perc1) & (cum / cum[-1] <= perc2)
            useind = sortind[ind]
            bin_index[useind] = int(ii + 1)

        if self.config.object_id_col != "":
            # below is commented out and replaced by a redundant line
            # because the data doesn't have ID yet
            # obj_id = test_data[self.config.object_id_col]
            obj_id = np.arange(npdf)
        else:
            # ID set to row index
            obj_id = np.arange(npdf)
            self.config.object_id_col = "row_index"

        class_id = {self.config.object_id_col: obj_id, "class_id": bin_index}
        self.add_data("output", class_id)
