from typing import Any

def dsps_population_sed_modeler(**kwargs) -> Any:
    """
    Derived class of Modeler for creating a population of galaxy rest-frame SED models
    using DSPS v3. (Hearin+21).
    SPS calculations are based on a set of template SEDs of simple stellar populations
    (SSPs).
    Supplying such templates is outside the planned scope of the DSPS package, and so
    they
    will need to be retrieved from some other library. For example, the FSPS library
    supplies
    such templates in a convenient form.

    The input galaxy properties, such as star-formation histories and metallicities,
    need to be supplied via an
    hdf5 table.

    The user-provided metallicity grid should be consistently defined with the
    metallicity of the templates SEDs.
    Users should be cautious in the use of the cosmic time grid. The time resolution
    strongly depends on the
    user scientific aim.
    jax serially execute the computations on CPU on single core, for CPU parallelization
    you need MPI.
    If GPU is used, jax natively and automatically parallelize the execution.

    ---

    This function generates the rest-frame SEDs and stores them into the Hdf5Handle.

    ---

    This function was generated from the function
    rail.creation.engines.dsps_sed_modeler.DSPSPopulationSedModeler.fit_model

    Parameters
    ----------
    input_data : str, required
        Filepath to the hdf5 table containing galaxy properties.
    Om0 : float, optional
        Omega matter: density of non-relativistic matter in units of the critical
        density at z=0.
    w0 : float, optional
        Dark energy equation of state at z=0 (a=1). This is pressure/density for dark
        energy in units where c=1.
    wa : float, optional
        Negative derivative of the dark energy equation of state with respect to the
        scale factor.
        A cosmological constant has w0=-1.0 and wa=0.0.
    h : float, optional
        dimensionless Hubble constant at z=0.
    ssp_templates_file : str, optional
        hdf5 file storing the SSP libraries used to create SEDs
        Default: rail_dsps/src/rail/examples_data/creation_data/data/dsps_default_data/s
        sp_data_fsps_v3.2_lgmet_age.h5
    redshift_key : str, optional
        Redshift keyword name of the hdf5 dataset
        Default: redshift
    cosmic_time_grid_key : str, optional
        Cosmic time grid keyword name of the hdf5 dataset, this is the grid of Universe
        age over which the stellar mass build-up takes place in units of Gyr
        Default: cosmic_time_grid
    star_formation_history_key : str, optional
        Star-formation history keyword name of the hdf5 dataset, this is the star-
        formation history of the galaxy in units of Msun/yr
        Default: star_formation_history
    stellar_metallicity_key : str, optional
        Stellar metallicity keyword name of the hdf5 dataset, this is the stellar
        metallicity in units of log10(Z)
        Default: stellar_metallicity
    stellar_metallicity_scatter_key : str, optional
        Stellar metallicity scatter keyword name of the hdf5 dataset, this is lognormal
        scatter in the metallicity distribution function
        Default: stellar_metallicity_scatter
    restframe_sed_key : str, optional
        Rest-frame SED keyword name of the output hdf5 dataset
        Default: restframe_seds
    default_cosmology : bool, optional
        True to use default DSPS cosmology. If False,Om0, w0, wa, h need to be supplied
        in the fit_model function
        Default: True
    min_wavelength : float, optional
        Minimum output rest-frame wavelength
        Default: 250
    max_wavelength : float, optional
        Maximum output rest-frame wavelength
        Default: 12000

    Returns
    -------
    dict
        Hdf5 table storing the rest-frame SED model
    """

def dsps_single_sed_modeler(**kwargs) -> Any:
    """
    Derived class of Modeler for creating a single galaxy rest-frame SED model using
    DSPS v3. (Hearin+21).
    SPS calculations are based on a set of template SEDs of simple stellar populations
    (SSPs).
    Supplying such templates is outside the planned scope of the DSPS package, and so
    they
    will need to be retrieved from some other library. For example, the FSPS library
    supplies
    such templates in a convenient form.

    The input galaxy properties, such as star-formation histories and metallicities,
    need to be supplied via an
    hdf5 table.

    The user-provided metallicity grid should be consistently defined with the
    metallicity of the templates SEDs.
    Users should be cautious in the use of the cosmic time grid. The time resolution
    strongly depends on the
    user scientific aim.

    ---

    This function generates the rest-frame SEDs and stores them into the Hdf5Handle.

    ---

    This function was generated from the function
    rail.creation.engines.dsps_sed_modeler.DSPSSingleSedModeler.fit_model

    Parameters
    ----------
    input_data : str, required
        Filepath to the hdf5 table containing galaxy properties.
    Om0 : float, optional
        Omega matter: density of non-relativistic matter in units of the critical
        density at z=0.
    w0 : float, optional
        Dark energy equation of state at z=0 (a=1). This is pressure/density for dark
        energy in units where c=1.
    wa : float, optional
        Negative derivative of the dark energy equation of state with respect to the
        scale factor.
        A cosmological constant has w0=-1.0 and wa=0.0.
    h : float, optional
        dimensionless Hubble constant at z=0.
    ssp_templates_file : str, optional
        hdf5 file storing the SSP libraries used to create SEDs
        Default: rail_dsps/src/rail/examples_data/creation_data/data/dsps_default_data/s
        sp_data_fsps_v3.2_lgmet_age.h5
    redshift_key : str, optional
        Redshift keyword name of the hdf5 dataset
        Default: redshifts
    cosmic_time_grid_key : str, optional
        Cosmic time grid keyword name of the hdf5 dataset, this is the grid of Universe
        age over which the stellar mass build-up takes place in units of Gyr
        Default: cosmic_time_grid
    star_formation_history_key : str, optional
        Star-formation history keyword name of the hdf5 dataset, this is the star-
        formation history of the galaxy in units of Msun/yr
        Default: star_formation_history
    stellar_metallicity_key : str, optional
        Stellar metallicity keyword name of the hdf5 dataset, this is the stellar
        metallicity in units of log10(Z)
        Default: stellar_metallicity
    stellar_metallicity_scatter_key : str, optional
        Stellar metallicity scatter keyword name of the hdf5 dataset, this is lognormal
        scatter in the metallicity distribution function
        Default: stellar_metallicity_scatter
    restframe_sed_key : str, optional
        Rest-frame SED keyword name of the output hdf5 dataset
        Default: restframe_sed
    default_cosmology : bool, optional
        True to use default DSPS cosmology. If False,Om0, w0, wa, h need to be supplied
        in the fit_model function
        Default: True
    min_wavelength : float, optional
        Minimum output rest-frame wavelength
        Default: 250
    max_wavelength : float, optional
        Maximum output rest-frame wavelength
        Default: 12000

    Returns
    -------
    dict
        Hdf5 table storing the rest-frame SED model
    """
