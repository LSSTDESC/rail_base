from typing import Any

def diff_sky_galaxy_population_modeler(**kwargs) -> Any:
    """
    Derived class of Modeler for creating a mock galaxy population using diffsky/skysim
    library.
    This class in particular samples the population parameters from the input
    diffsky/skysim catalog.

    ---

    This function samples the population parameters from the diffsky/skysim galaxy
    population model and stores
    them into an Hdf5Handle.

    ---

    This function was generated from the function rail.creation.engines.galaxy_populatio
    n_components_modeler.DiffskyGalaxyPopulationModeler.fit_model

    Parameters
    ----------
    input_data : str
        This is the input diffsky/skysim catalog path.
    diffmah_keys : list, optional
        Keywords list in the skysim/diffsky catalog that store diffmah parameters.
        Default: ['diffmah_logmp_fit', 'diffmah_mah_logtc', 'diffmah_early_index',...]
    diffstar_ms_keys : list, optional
        Keywords list in the skysim/diffsky catalog that store diffstar main sequence
        parameters.
        Default: ['diffstar_u_lgmcrit', 'diffstar_u_lgy_at_mcrit',...]
    diffstar_q_keys : list, optional
        Keywords list in the skysim/diffsky catalog that store diffstar quenching
        parameters.
        Default: ['diffstar_u_qt', 'diffstar_u_qs', 'diffstar_u_q_drop',...]
    catalog_redshift_key : str, optional
        Redshift keyword in the skysim/diffsky catalog.
        Default: redshift
    catalog_metallicity_key : str, optional
        Stellar metallicity keyword in the skysim/diffsky catalog.
        Default: lg_met_mean
    catalog_metallicity_scatter_key : str, optional
        Stellar metallicity scatter keyword in the skysim/diffsky catalog.
        Default: lg_met_scatter

    Returns
    -------
    dict
        Hdf5 table storing the population parameters.
    """
