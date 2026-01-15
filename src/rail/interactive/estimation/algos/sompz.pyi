from typing import Any

def sompz_estimator(**kwargs) -> Any:
    """
    CatEstimator subclass to compute redshift PDFs for SOMPZ

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZEstimator.estimate

    Parameters
    ----------
    input : dict["spec_data": Any, "balrog_data": Any, "wide_data": Any]
        Dictionary of input data with the following keys:
        spec_data: Any - description
        balrog_data: Any - description
        wide_data: Any - description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    bin_edges : list, optional
        list of edges of tomo bins
        Default: [0.0, 0.405, 0.665, 0.96, 2.0]
    zbins_min : float, optional
        minimum redshift for output grid
        Default: 0.0
    zbins_max : float, optional
        maximum redshift for output grid
        Default: 6.0
    zbins_dz : float, optional
        delta z for defining output grid
        Default: 0.01
    spec_groupname : str, optional
        hdf5_groupname for spec_data
        Default: photometry
    balrog_groupname : str, optional
        hdf5_groupname for balrog_data
        Default: photometry
    wide_groupname : str, optional
        hdf5_groupname for wide_data
        Default: photometry
    specz_name : str, optional
        column name for true redshift in specz sample
        Default: redshift
    inputs_deep : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    input_errs_deep : list, optional
        list of the names of columns containing errors on inputs for deep data
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    inputs_wide : list, optional
        list of the names of columns to be used as inputs for wide data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    input_errs_wide : list, optional
        list of the names of columns containing errors on inputs for wide data
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    zero_points_deep : list, optional
        zero points for converting mags to fluxes for deep data, if needed
        Default: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    zero_points_wide : list, optional
        zero points for converting mags to fluxes for wide data, if needed
        Default: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    som_shape_deep : list, optional
        shape for the deep som, must be a 2-element tuple
        Default: [32, 32]
    som_shape_wide : list, optional
        shape for the wide som, must be a 2-element tuple
        Default: [32, 32]
    som_minerror_deep : float, optional
        floor placed on observational error on each feature in deep som
        Default: 0.01
    som_minerror_wide : float, optional
        floor placed on observational error on each feature in wide som
        Default: 0.01
    som_wrap_deep : bool, optional
        flag to set whether the deep SOM has periodic boundary conditions
        Default: False
    som_wrap_wide : bool, optional
        flag to set whether the wide SOM has periodic boundary conditions
        Default: False
    som_take_log_deep : bool, optional
        flag to set whether to take log of inputs (i.e. for fluxes) for deep som
        Default: True
    som_take_log_wide : bool, optional
        flag to set whether to take log of inputs (i.e. for fluxes) for wide som
        Default: True
    convert_to_flux_deep : bool, optional
        flag for whether to convert input columns to fluxes for deep dataset to true if
        inputs are mags and to False if inputs are already fluxes
        Default: False
    convert_to_flux_wide : bool, optional
        flag for whether to convert input columns to fluxes for wide data
        Default: False
    set_threshold_deep : bool, optional
        flag for whether to replace values below a threshold with a set number
        Default: False
    thresh_val_deep : float, optional
        threshold value for set_threshold for deep data
        Default: 1e-05
    set_threshold_wide : bool, optional
        flag for whether to replace values below a threshold with a set number
        Default: False
    thresh_val_wide : float, optional
        threshold value for set_threshold for wide data
        Default: 1e-05
    debug : bool, optional
        boolean reducing dataset size for quick debuggin
        Default: False

    Returns
    -------
    None
    """

def sompz_estimator_base(**kwargs) -> Any:
    """
    CatEstimator subclass to compute redshift PDFs for SOMPZ

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZEstimatorBase.estimate

    Parameters
    ----------
    input : Any
        description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    inputs : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    input_errs : list, optional
        list of the names of columns containing errors on inputs for deep data
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    zero_points : list, optional
        zero points for converting mags to fluxes for deep data, if needed
        Default: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    som_shape : list, optional
        shape for the deep som, must be a 2-element tuple
        Default: [32, 32]
    som_minerror : float, optional
        floor placed on observational error on each feature in deep som
        Default: 0.01
    som_wrap : bool, optional
        flag to set whether the deep SOM has periodic boundary conditions
        Default: False
    som_take_log : bool, optional
        flag to set whether to take log of inputs (i.e. for fluxes) for deep som
        Default: True
    convert_to_flux : bool, optional
        flag for whether to convert input columns to fluxes for deep dataset to true if
        inputs are mags and to False if inputs are already fluxes
        Default: False
    set_threshold : bool, optional
        flag for whether to replace values below a threshold with a set number
        Default: False
    thresh_val : float, optional
        threshold value for set_threshold for deep data
        Default: 1e-05
    debug : bool, optional
        boolean reducing dataset size for quick debuggin
        Default: False

    Returns
    -------
    None
    """

def sompz_estimator_deep(**kwargs) -> Any:
    """
    CatEstimator subclass to compute redshift PDFs for SOMPZ

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZEstimatorDeep.estimate

    Parameters
    ----------
    input : Any
        description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    inputs : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    input_errs : list, optional
        list of the names of columns containing errors on inputs for deep data
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    zero_points : list, optional
        zero points for converting mags to fluxes for deep data, if needed
        Default: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    som_shape : list, optional
        shape for the deep som, must be a 2-element tuple
        Default: [32, 32]
    som_minerror : float, optional
        floor placed on observational error on each feature in deep som
        Default: 0.01
    som_wrap : bool, optional
        flag to set whether the deep SOM has periodic boundary conditions
        Default: False
    som_take_log : bool, optional
        flag to set whether to take log of inputs (i.e. for fluxes) for deep som
        Default: True
    convert_to_flux : bool, optional
        flag for whether to convert input columns to fluxes for deep dataset to true if
        inputs are mags and to False if inputs are already fluxes
        Default: False
    set_threshold : bool, optional
        flag for whether to replace values below a threshold with a set number
        Default: False
    thresh_val : float, optional
        threshold value for set_threshold for deep data
        Default: 1e-05
    debug : bool, optional
        boolean reducing dataset size for quick debuggin
        Default: False

    Returns
    -------
    None
    """

def sompz_estimator_wide(**kwargs) -> Any:
    """
    CatEstimator subclass to compute redshift PDFs for SOMPZ

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZEstimatorWide.estimate

    Parameters
    ----------
    input : Any
        description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    inputs : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    input_errs : list, optional
        list of the names of columns containing errors on inputs for deep data
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    zero_points : list, optional
        zero points for converting mags to fluxes for deep data, if needed
        Default: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    som_shape : list, optional
        shape for the deep som, must be a 2-element tuple
        Default: [32, 32]
    som_minerror : float, optional
        floor placed on observational error on each feature in deep som
        Default: 0.01
    som_wrap : bool, optional
        flag to set whether the deep SOM has periodic boundary conditions
        Default: False
    som_take_log : bool, optional
        flag to set whether to take log of inputs (i.e. for fluxes) for deep som
        Default: True
    convert_to_flux : bool, optional
        flag for whether to convert input columns to fluxes for deep dataset to true if
        inputs are mags and to False if inputs are already fluxes
        Default: False
    set_threshold : bool, optional
        flag for whether to replace values below a threshold with a set number
        Default: False
    thresh_val : float, optional
        threshold value for set_threshold for deep data
        Default: 1e-05
    debug : bool, optional
        boolean reducing dataset size for quick debuggin
        Default: False

    Returns
    -------
    None
    """

def sompz_informer(**kwargs) -> Any:
    """
    Inform stage for SOMPZEstimator

    ---

    Inform description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZInformer.inform

    Parameters
    ----------
    input : Any
        description
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    nproc : int, optional
        number of processors to use
        Default: 1
    inputs : list, optional
        list of the names of columns to be used as inputs for data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    input_errs : list, optional
        list of the names of columns containing errors on inputs for data
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    zero_points : list, optional
        zero points for converting mags to fluxes for data, if needed
        Default: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    som_shape : list, optional
        shape for the som, must be a 2-element tuple
        Default: [32, 32]
    som_minerror : float, optional
        floor placed on observational error on each feature in som
        Default: 0.01
    som_wrap : bool, optional
        flag to set whether the SOM has periodic boundary conditions
        Default: False
    som_take_log : bool, optional
        flag to set whether to take log of inputs (i.e. for fluxes) for som
        Default: True
    convert_to_flux : bool, optional
        flag for whether to convert input columns to fluxes for dataset to true if
        inputs are mags and to False if inputs are already fluxes
        Default: False
    set_threshold : bool, optional
        flag for whether to replace values below a threshold with a set number
        Default: False
    thresh_val : float, optional
        threshold value for set_threshold for data
        Default: 1e-05
    thresh_val_err : float, optional
        threshold value for set_threshold for data error
        Default: 1e-05

    Returns
    -------
    None
        self.model looks like it's never set
    """

def sompz_p_c_chat(**kwargs) -> Any:
    """
    Calcaulate p(c|chat)

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZPc_chat.estimate

    Parameters
    ----------
    input : dict["cell_deep_balrog_data": Any, "cell_wide_balrog_data": Any]
        Dictionary of input data with the following keys:
        cell_deep_balrog_data: Any - description
        cell_wide_balrog_data: Any - description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    inputs : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]

    Returns
    -------
    None
    """

def sompz_pzc(**kwargs) -> Any:
    """
    Calcaulate pzc

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZPzc.estimate

    Parameters
    ----------
    input : dict["spec_data": Any, "cell_deep_spec_data": Any]
        Dictionary of input data with the following keys:
        spec_data: Any - description
        cell_deep_spec_data: Any - description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    inputs : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    deep_groupname : str, optional
        hdf5_groupname for deep file
        Default: photometry
    bin_edges : list, optional
        list of edges of tomo bins
        Default: [0.0, 0.405, 0.665, 0.96, 2.0]
    zbins_min : float, optional
        minimum redshift for output grid
        Default: 0.0
    zbins_max : float, optional
        maximum redshift for output grid
        Default: 6.0
    zbins_dz : float, optional
        delta z for defining output grid
        Default: 0.01

    Returns
    -------
    None
    """

def sompz_pzchat(**kwargs) -> Any:
    """
    Calcaulate pzchat

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZPzchat.estimate

    Parameters
    ----------
    input : dict["spec_data": Any, "cell_deep_spec_data": Any, "cell_wide_wide_data":
    Any, "pz_c": Any, "pc_chat": Any]
        Dictionary of input data with the following keys:
        spec_data: Any - description
        cell_deep_spec_data: Any - description
        cell_wide_wide_data: Any - description
        pz_c: Any - description
        pc_chat: Any - description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    inputs : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    bin_edges : list, optional
        list of edges of tomo bins
        Default: [0.0, 0.405, 0.665, 0.96, 2.0]
    zbins_min : float, optional
        minimum redshift for output grid
        Default: 0.0
    zbins_max : float, optional
        maximum redshift for output grid
        Default: 6.0
    zbins_dz : float, optional
        delta z for defining output grid
        Default: 0.01

    Returns
    -------
    None
    """

def sompz_tomobin(**kwargs) -> Any:
    """
    Calcaulate tomobin

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZTomobin.estimate

    Parameters
    ----------
    input : dict["spec_data": Any, "cell_deep_spec_data": Any, "cell_wide_spec_data":
    Any]
        Dictionary of input data with the following keys:
        spec_data: Any - description
        cell_deep_spec_data: Any - description
        cell_wide_spec_data: Any - description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    inputs : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    bin_edges : list, optional
        list of edges of tomo bins
        Default: [0.0, 0.405, 0.665, 0.96, 2.0]
    zbins_min : float, optional
        minimum redshift for output grid
        Default: 0.0
    zbins_max : float, optional
        maximum redshift for output grid
        Default: 6.0
    zbins_dz : float, optional
        delta z for defining output grid
        Default: 0.01

    Returns
    -------
    None
    """

def sompz_nz(**kwargs) -> Any:
    """
    Calcaulate nz

    ---

    Estimate description

    ---

    This function was generated from the function
    rail.estimation.algos.sompz.SOMPZnz.estimate

    Parameters
    ----------
    input : dict["spec_data": Any, "cell_deep_spec_data": Any, "cell_wide_wide_data":
    Any, "tomo_bins_wide": Any, "pc_chat": Any]
        Dictionary of input data with the following keys:
        spec_data: Any - description
        cell_deep_spec_data: Any - description
        cell_wide_wide_data: Any - description
        tomo_bins_wide: Any - description
        pc_chat: Any - description
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    inputs : list, optional
        list of the names of columns to be used as inputs for deep data
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    bin_edges : list, optional
        list of edges of tomo bins
        Default: [0.0, 0.405, 0.665, 0.96, 2.0]
    zbins_min : float, optional
        minimum redshift for output grid
        Default: 0.0
    zbins_max : float, optional
        maximum redshift for output grid
        Default: 6.0
    zbins_dz : float, optional
        delta z for defining output grid
        Default: 0.01

    Returns
    -------
    None
    """
