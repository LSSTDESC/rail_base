from typing import Any

def fsps_photometry_creator(**kwargs) -> Any:
    """
    Derived class of Creator that generate synthetic photometric fsps_default_data from
    the rest-frame SED model
    generated with the FSPSSedModeler class.
    The user is required to provide galaxy redshifts and filter information in an .npy
    format for the code to run.
    The restframe SEDs are stored in a pickle file or passed as ModelHandle.
    Details of what each file should contain are explicited in config_options.
    The output is a Fits table containing magnitudes.

    ---

    Creates observed magnitudes for the population of galaxies and stores them into an
    Hdf5Handle.

    ---

    This function was generated from the function
    rail.creation.engines.fsps_photometry_creator.FSPSPhotometryCreator.sample

    Parameters
    ----------
    input_data : dict, required
        Hdf5Handle containing the rest-frame SED models.
    model : dict, required
    seed : int | None, optional
        The random seed to control sampling, by default None
    redshift_key : str, optional
        Redshift keyword name of the hdf5 dataset containing rest-frame SEDs
        Default: redshifts
    restframe_sed_key : str, optional
        Rest-frame SED keyword name of the hdf5 dataset containing rest-frame SEDs
        Default: restframe_seds
    restframe_wave_key : str, optional
        Rest-frame wavelengths keyword name of thehdf5 dataset containing rest-frame
        SEDs
        Default: wavelength
    apparent_mags_key : str, optional
        Apparent magnitudes keyword name of the output hdf5 dataset
        Default: apparent_mags
    filter_folder : str, optional
        Folder containing filter transmissions
        Default: rail/examples_data/creation_data/data/fsps_default_data/filters
    instrument_name : str, optional
        Instrument name as prefix to filter transmission files
        Default: lsst
    wavebands : str, optional
        Comma-separated list of wavebands
        Default: u,g,r,i,z,y
    filter_wave_key : str, optional
        Default: wave
    filter_transm_key : str, optional
        Default: transmission
    Om0 : float, optional
        Omega matter at current time
        Default: 0.3
    Ode0 : float, optional
        Omega dark energy at current time
        Default: 0.7
    w0 : float, optional
        Dark energy equation-of-state parameter at current time
        Default: -1
    wa : float, optional
        Slope dark energy equation-of-state evolution with scale factor
        Default: 0.0
    h : float, optional
        Dimensionless hubble constant
        Default: 0.7
    use_planck_cosmology : bool, optional
        True to overwrite the cosmological parameters to their Planck2015 values
        Default: False
    physical_units : bool, optional
        A parameter
        Default: False
    msg : unknown type, optional
        Default: False (True) for rest-frame spectra in units ofLsun/Hz (erg/s/Hz)

    Returns
    -------
    dict
        Hdf5Handle storing the apparent magnitudes and redshifts of galaxies.


    Notes
    -----
    This method puts  `seed` into the stage configuration data, which makes them
    available to other methods.
    It then calls the `run` method. Finally, the `Hdf5Handle` associated to the `output`
    tag is returned.
    """
