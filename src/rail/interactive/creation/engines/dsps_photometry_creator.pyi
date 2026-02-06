from typing import Any

def dsps_photometry_creator(**kwargs) -> Any:
    """
    Derived class of Creator that generate synthetic absolute and apparent magnitudes
    from one or more SED models generated with the DSPSSingleSedModeler or
    DSPSPopulationSedModeler classes.
    It accepts as input Hdf5Handles containing the rest-frame SEDs in units of Lsun/Hz
    and outputs an Hdf5Handle
    containing sequential indices, absolute and apparent magnitudes for each galaxy.
    Photometric quantities are computed for the filters defined in the configuration
    file.

    jax serially execute the computations on CPU on single core, for CPU parallelization
    you need MPI.
    If GPU is used, jax natively and automatically parallelize the execution.

    ---

    Creates observed and absolute magnitudes for the population of galaxy rest-frame
    SEDs and stores them into
    an Hdf5Handle.

    ---

    This function was generated from the function
    rail.creation.engines.dsps_photometry_creator.DSPSPhotometryCreator.sample

    Parameters
    ----------
    input_data : str, required
        Filepath to the hdf5 table containing the galaxy rest-frame SEDs.
    model : dict, required
    seed : int, optional
        The random seed to control sampling
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
    redshift_key : str, optional
        Redshift keyword name of the hdf5 dataset containing rest-frame SEDs
        Default: redshifts
    restframe_sed_key : str, optional
        Rest-frame SED keyword name of the hdf5 dataset containing rest-frame SEDs
        Default: restframe_seds
    absolute_mags_key : str, optional
        Absolute magnitudes keyword name of the output hdf5 dataset
        Default: rest_frame_absolute_mags
    apparent_mags_key : str, optional
        Apparent magnitudes keyword name of the output hdf5 dataset
        Default: apparent_mags
    filter_folder : str, optional
        Folder containing filter transmissions
        Default:
        rail_dsps/src/rail/examples_data/creation_data/data/dsps_default_data/filters
    instrument_name : str, optional
        Instrument name as prefix to filter transmission files
        Default: lsst
    wavebands : str, optional
        Comma-separated list of wavebands
        Default: u,g,r,i,z,y
    min_wavelength : float, optional
        Minimum input rest-frame wavelength SEDs
        Default: 250
    max_wavelength : float, optional
        Maximum input rest-frame wavelength SEDs
        Default: 12000
    ssp_templates_file : str, optional
        hdf5 file storing the SSP libraries used to create SEDs
        Default: rail_dsps/src/rail/examples_data/creation_data/data/dsps_default_data/s
        sp_data_fsps_v3.2_lgmet_age.h5
    default_cosmology : bool, optional
        True to use default DSPS cosmology. If False,Om0, w0, wa, h need to be supplied
        in the sample function
        Default: True

    Returns
    -------
    dict
        Hdf5Handle storing the absolute and apparent magnitudes.


    Notes
    -----
    This method puts  `seed` into the stage configuration data, which makes them
    available to other methods.
    It then calls the `run` method. Finally, the `Hdf5Handle` associated to the `output`
    tag is returned.
    """
