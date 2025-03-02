import numpy as np
import scipy.stats as stats
from rail.core.stage import RailStage
from rail.core.data import TableHandle, Hdf5Handle
from rail.evaluation.evaluator import Evaluator
from typing import Any
from rail.core.common_params import SHARED_PARAMS
from ceci.config import StageParameter as Param


class KDEBinOverlap(RailStage):
    name = "KDEBinOverlap"
    inputs = [("truth", TableHandle), ("bin_index", Hdf5Handle)]
    outputs = [("output", Hdf5Handle)]
    
    config_options = RailStage.config_options.copy()
    config_options.update(
        hdf5_groupname=Param(
            str, "", required=False, msg="HDF5 Groupname for truth table."
        ),
        redshift_col=SHARED_PARAMS
    )
    # metric_base_class = Evaluator
    
    def __init__(self, args: Any, **kwargs: Any) -> None:
        
        super().__init__(args, **kwargs)

    
    def evaluate(self, bin_index, truth):
        
        self.set_data("bin_index", bin_index)
        self.set_data("truth", truth)
        
        self.run()
        return self.get_handle("output")
    
    
    def run(self):
        true_redshifts = self.get_handle("truth").data[self.config.hdf5_groupname][self.config.redshift_col]  # 1D array of redshifts
        bin_indices = self.get_handle("bin_index").data['class_id']  # 1D array of bin indices

        unique_bins = np.unique(bin_indices)
        N = len(unique_bins)
        overlap_matrix = np.zeros((N, N))

        kde_dict = {}
        for bin_id in unique_bins:
            redshifts_in_bin = true_redshifts[bin_indices == bin_id]
            if len(redshifts_in_bin) > 1:
                kde = stats.gaussian_kde(redshifts_in_bin)
                kde_dict[bin_id] = kde

        for i, bin_i in enumerate(unique_bins):
            overlap_matrix[i][i] = 1.0
            for j, bin_j in enumerate(unique_bins):
                if j >= i:
                    continue
                
                kde_i = kde_dict.get(bin_i, None)
                kde_j = kde_dict.get(bin_j, None)
                
                if kde_i is not None and kde_j is not None:
                    # Define the evaluation grid based on the union of both bin samples
                    eval_grid = np.linspace(min(true_redshifts), max(true_redshifts), 1000)
                    p_i = kde_i(eval_grid)
                    p_j = kde_j(eval_grid)
                    
                    # Compute overlap as the integral of the minimum of both distributions
                    overlap = np.trapz(np.minimum(p_i, p_j), eval_grid)
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap  # Symmetric matrix

        self.add_handle("output", data = overlap_matrix)
