"""
A classifier that uses pz point estimate to assign
tomographic bins with uniform binning. 
"""

import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.classifier import PZClassifier
from rail.core.data import TableHandle


class UniformBinningClassifier(PZClassifier):
    """Classifier that simply assign tomographic
    bins based on point estimate according to SRD"""

    name = "UniformBinningClassifier"
    config_options = PZClassifier.config_options.copy()
    config_options.update(
        id_name=Param(
            str,
            "",
            msg="Column name for the object ID in the input data, if empty the row index is used as the ID.",
        ),
        point_estimate=Param(str, "zmode", msg="Which point estimate to use"),
        zbin_edges=Param(
            list,
            [],
            msg="The tomographic redshift bin edges."
            "If this is given (contains two or more entries), all settings below will be ignored.",
        ),
        zmin=Param(float, 0.0, msg="Minimum redshift of the sample"),
        zmax=Param(float, 3.0, msg="Maximum redshift of the sample"),
        nbins=Param(int, 5, msg="Number of tomographic bins"),
        no_assign=Param(int, -99, msg="Value for no assignment flag"),
    )
    outputs = [("output", TableHandle)]

    def __init__(self, args, comm=None):
        PZClassifier.__init__(self, args, comm=comm)

    def run(self):
        test_data = self.get_data("input")
        npdf = test_data.npdf

        try:
            zb = test_data.ancil[self.config.point_estimate]
        except KeyError as msg:
            raise KeyError(
                f"{self.config.point_estimate} is not contained in the data ancil, "
                "you will need to compute it explicitly."
            ) from msg

        # binning options
        if len(self.config.zbin_edges) >= 2:
            # this overwrites all other key words
            # linear binning defined by zmin, zmax, and nbins
            bin_index = np.digitize(zb, self.config.zbin_edges)
            # assign -99 to objects not in any bin:
            bin_index[bin_index == 0] = self.config.no_assign
            bin_index[bin_index == len(self.config.zbin_edges)] = self.config.no_assign

        else:
            # linear binning defined by zmin, zmax, and nbins
            bin_index = np.digitize(
                zb,
                np.linspace(self.config.zmin, self.config.zmax, self.config.nbins + 1),
            )
            # assign -99 to objects not in any bin:
            bin_index[bin_index == 0] = self.config.no_assign
            bin_index[bin_index == (self.config.nbins + 1)] = self.config.no_assign

        if self.config.id_name != "":
            # below is commented out and replaced by a redundant line
            # because the data doesn't have ID yet
            # obj_id = test_data[self.config.id_name]
            obj_id = np.arange(npdf)
        elif self.config.id_name == "":
            # ID set to row index
            obj_id = np.arange(npdf)
            self.config.id_name = "row_index"

        class_id = {self.config.id_name: obj_id, "class_id": bin_index}
        self.add_data("output", class_id)
