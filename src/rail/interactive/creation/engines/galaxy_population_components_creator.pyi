from typing import Any

def diff_sky_galaxy_population_creator(**kwargs) -> Any:
    """
    Derived class of Creator that samples galaxy properties from population parameters
    generated with
    DiffskyGalaxyPopulationModeler.

    ---

    Samples galaxy properties from diffsky/skysim model and stores them into an
    Hdf5Handle

    ---

    This function was generated from the function rail.creation.engines.galaxy_populatio
    n_components_creator.DiffskyGalaxyPopulationCreator.sample

    Parameters
    ----------
    seed : int
        The random seed to control sampling
    input_data : Hdf5Handle
        This is the input diffsky/skysim population properties catalog path.
    log10_age_universe : float, optional
        Base-10 log of the age of the universe in Gyr.
        Default: 1.1398790864012365
    cosmic_baryon_fraction : float, optional
        Cosmic baryon fraction.
        Default: 0.156
    t_min_table : float, optional
        Lower value of the Universe age time grid.
        Default: 0.01
    t_max_table : float, optional
        Upper value of the Universe age time grid.
        Default: 13.799999999999999
    n_time_steps : int, optional
        Number of steps of the Universe age time grid.
        Default: 50
    tacc_integration_min : float, optional
        Earliest time to use in the tacc integrations. Default is 0.01 Gyr.
        Default: 0.001
    cosmology_parameters : tuple, optional
        NamedTuple storing parameters of a flat w0-wa cdm cosmology, default is
        Planck15.
        Default: CosmoParams(Om0=0.3075, w0=-1.0, wa=0.0, h=0.6774)
    catalog_redshift_key : str, optional
        Redshift keyword in the skysim/diffsky catalog.
        Default: redshift
    catalog_metallicity_key : str, optional
        Stellar metallicity keyword in the skysim/diffsky catalog.
        Default: lg_met_mean
    catalog_metallicity_scatter_key : str, optional
        Stellar metallicity scatter keyword in the skysim/diffsky catalog.
        Default: lg_met_scatter
    cosmic_time_grid_key : str, optional
        Cosmic time grid keyword in the output catalog.
        Default: cosmic_time_grid
    star_formation_history_key : str, optional
        Star-formation history keyword in the output catalog.
        Default: star_formation_history
    star_formation_rate_key : str, optional
        Star-formation rate keyword in the output catalog.
        Default: star_formation_rate
    stellar_mass_history_key : str, optional
        Stellar mass history keyword in the output catalog.
        Default: stellar_mass_history
    stellar_mass_key : str, optional
        Stellar mass keyword in the output catalog.
        Default: stellar_mass

    Returns
    -------
    dict
        Hdf5 Handle storing the sampled galaxy properties.


    Notes
    -----
    This method puts  `seed` into the stage configuration data, which makes them
    available to other methods.
    It then calls the `run` method. Finally, the `Hdf5Handle` associated to the `output`
    tag is returned.
    """
