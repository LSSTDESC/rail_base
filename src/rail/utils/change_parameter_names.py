"""Script to change the parameter names in older yaml pipeline files to match RAIL v2.0 parameter name updates."""

import yaml

# dictionary of parameters from rail v1.2 as keys, with their new names in v1.3 (or 2.0) as the corresponding value
parameter_change_dict = {
    # rail_base
    "_random_state": "seed",
    "inplace": "in_place",
    "limits": "metric_integration_limits",
    "nbins": "n_tom_bins",
    "niter": "n_iter",
    "nsamples": "n_samples",
    "num_samples": "n_samples",
    "point_estimate": "point_estimate_key",
    "start": "start_row",
    "stop": "stop_row",
    "id_name": "object_id_col",
    "rand_zmax": "zmax",
    "rand_zmin": "zmin",
    # rail_astro_tools
    "copy_cols": "copy_col_dict",
    "error_columns": "err_bands",
    "N_tot": "n_tot",
    "random_seed": "seed",
    # pzflow
    "error_names_dict": "err_names_dict",
    "flow_seed": "seed",
    "num_training_epochs": "n_training_epochs",
    "phys_cols": None,  # to be figured out
    "redshift_column_name": "redshift_col",
    "ref_column_name": "ref_band",
    # rail sklearn
    "bands": "band_dict",  # or band_map
    "id_name": "object_id_col",
    "phot_weightcol": "phot_weight_col",
    "szname": "sz_name",
    "trainfrac": "train_frac",
    # delight
    "bands_names": "filter_list",
    "dlght_redshiftBinSize": "dz",
    "dlght_redshiftMax": "zmax",
    "dlght_redshiftMin": "zmin",
    "lineWidthSigma": "line_width_sigma",
    "zPriorSigma": "z_prior_sigma",
    # bpz
    "data_path": "bpz_reference_data_path",
    "unobserved_val": "nonobserved_val",
    # fsps
    "dell": "del_l",
    "delt": "del_t",
    "h": "cosmo_h",
    "zmet_key": "Z_met_key",
    # som
    "gridtype": "grid_type",
    "redshift_colname": "redshift_col",
    "split": "som_split",
    # tpz
    "minleaf": "min_leaf",
    "natt": "n_att",
    "nrandom": "n_random",
    "ntrees": "n_trees",
    # flexzboost
    "nsharp": "n_sharp",
}


# function for changing variables in _config.yml files
# IN PROGRESS
# def update_parameter_names_to_v2(input_param_file: str):

#     with open(input_param_file, encoding="utf-8") as pfile:
#         input_params_dict = yaml.safe_load(pfile)

#         for algo, params in input_params_dict.items():
#             for key, param in params.items():
#                 if key in parameter_change_dict.keys():
#                     # update the parameter in the dictionary
#                     pass
