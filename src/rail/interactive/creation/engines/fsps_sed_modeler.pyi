from typing import Any

def fsps_sed_modeler(**kwargs) -> Any:
    """
    Derived class of Modeler for creating a single galaxy rest-frame SED model using
    FSPS (Conroy08).

    Only the most important parameters are provided via config_options. The remaining
    ones from FSPS can be
    provided when creating the rest-frame SED model.

    Install FSPS with the following commands:

    .. code-block:: text

        pip uninstall fsps
       git clone --recursive https://github.com/dfm/python-fsps.git
       cd python-fsps
       python -m pip install .
       export SPS_HOME=$(pwd)/src/fsps/libfsps

    ---

    This function creates rest-frame SED models from an input galaxy population catalog.

    ---

    This function was generated from the function
    rail.creation.engines.fsps_sed_modeler.FSPSSedModeler.fit_model

    Parameters
    ----------
    input_data : dict, required
        This is the input catalog in the form of an Hdf5Handle.
    chunk_size : unknown type, optional
        Default: 10000
    hdf5_groupname : unknown type, optional
        Default: <class 'str'>
    compute_vega_mags : bool, optional
        True uses Vega magnitudes versus AB magnitudes
        Default: False
    vactoair_flag : bool, optional
        If True, output wavelengths in air (rather than vac)
        Default: False
    zcontinuous : int, optional
        Flag for interpolation in metallicity of SSP before CSP
        Default: 1
    add_agb_dust_model : bool, optional
        Turn on/off adding AGB circumstellar dust contribution to SED
        Default: True
    add_dust_emission : bool, optional
        Turn on/off adding dust emission contribution to SED
        Default: True
    add_igm_absorption : bool, optional
        Turn on/off adding IGM absorption contribution to SED
        Default: False
    add_neb_emission : bool, optional
        Turn on/off nebular emission model based on Cloudy
        Default: False
    add_neb_continuum : bool, optional
        Turn on/off nebular continuum component
        Default: False
    add_stellar_remnants : bool, optional
        Turn on/off adding stellar remnants contribution to stellar mass
        Default: True
    compute_light_ages : bool, optional
        If True then the returned spectra are actually light-weighted ages (in Gyr)
        Default: False
    nebemlineinspec : bool, optional
        True to include emission line fluxes in spectrum
        Default: False
    smooth_velocity : bool, optional
        True/False for smoothing in velocity/wavelength space
        Default: True
    smooth_lsf : bool, optional
        True/False for smoothing SSPs by a wavelength dependent line spread function
        Default: False
    cloudy_dust : bool, optional
        Switch to include dust in the Cloudy tables
        Default: False
    agb_dust : float, optional
        Scales the circumstellar AGB dust emission
        Default: 1.0
    tpagb_norm_type : int, optional
        Flag for TP-AGB normalization scheme, default Villaume, Conroy, Johnson 2015
        normalization
        Default: 2
    dell : float, optional
        Shift in log(L_bol) of the TP-AGB isochrones
        Default: 0.0
    delt : float, optional
        Shift in log(T_eff) of the TP-AGB isochrones
        Default: 0.0
    redgb : float, optional
        Modify weight given to RGB. Only available with BaSTI isochrone set
        Default: 1.0
    agb : float, optional
        Modify weight given to TP-AGB
        Default: 1.0
    fcstar : float, optional
        Fraction of stars that the Padova isochrones identify as Carbon stars
        Default: 1.0
    sbss : float, optional
        Specific frequency of blue straggler stars
        Default: 0.0
    fbhb : float, optional
        Fraction of horizontal branch stars that are blue
        Default: 0.0
    pagb : float, optional
        Weight given to the post–AGB phase
        Default: 1.0
    redshifts_key : str, optional
        galaxy redshift, dataset keyword name
        Default: redshifts
    zmet_key : str, optional
        The metallicity is specified as an integer ranging between 1 and nz. If
        zcontinuous > 0 then this parameter is ignored, dataset keyword name
        Default: zmet
    stellar_metallicities_key : str, optional
        galaxy stellar metallicities (log10(Z / Zsun)) dataset keyword name, to be used
        with zcontinuous > 0,dataset keyword name
        Default: stellar_metallicity
    pmetals_key : str, optional
        The power for the metallicty distribution function,only used if zcontinous=2,
        dataset keyword name
        Default: pmetals
    imf_type : int, optional
        IMF type, see FSPS manual, default Chabrier IMF
        Default: 1
    imf_upper_limit : float, optional
        The upper limit of the IMF in solar masses
        Default: 120.0
    imf_lower_limit : float, optional
        The lower limit of the IMF in solar masses
        Default: 0.08
    imf1 : float, optional
        log slope of IMF in 0.08<M/Msun<0.5, if imf_type=2
        Default: 1.3
    imf2 : float, optional
        log slope of IMF in 0.5<M/Msun<1, if imf_type=2
        Default: 2.3
    imf3 : float, optional
        log slope of IMF in M/Msun>1, if imf_type=2
        Default: 2.3
    vdmc : float, optional
        IMF parameter defined in van Dokkum (2008). Only used if imf_type=3
        Default: 0.08
    mdave : float, optional
        IMF parameter defined in Dave (2008). Only used if imf_type=4.
        Default: 0.5
    evtype : int, optional
        Compute SSPs for only the given evolutionary type. All phases used when set to
        -1.
        Default: -1
    use_wr_spectra : int, optional
        Turn on/off the WR spectral library
        Default: 1
    logt_wmb_hot : float, optional
        Use the Eldridge (2017) WMBasic hot star library above this value of log(T_eff)
        or 25,000K,whichever is larger
        Default: 0.0
    masscut : float, optional
        Truncate the IMF above this value
        Default: 150.0
    velocity_dispersions_key : str, optional
        stellar velocity dispersions (km/s), dataset keyword name
        Default: stellar_velocity_dispersion
    min_wavelength : float, optional
        minimum rest-frame wavelength
        Default: 3000
    max_wavelength : float, optional
        maximum rest-frame wavelength
        Default: 10000
    gas_ionizations_key : str, optional
        gas ionization values dataset keyword name
        Default: gas_ionization
    gas_metallicities_key : str, optional
        gas metallicities (log10(Zgas / Zsun)) dataset keyword name
        Default: gas_metallicity
    igm_factor : float, optional
        Factor used to scale the IGM optical depth
        Default: 1.0
    sfh_type : int, optional
        star-formation history type, see FSPS manual, default SSP
        Default: 0
    tau_key : str, optional
        Defines e-folding time for the SFH, in Gyr. Only used if sfh=1 or sfh=4, dataset
        keyword name
        Default: tau
    const_key : str, optional
        Defines the constant component of the SFH, Only used if sfh=1 or sfh=4, dataset
        keyword name
        Default: const
    sf_start_key : str, optional
        Start time of the SFH, in Gyr. Only used if sfh=1 or sfh=4 or sfh=5, dataset
        keyword name
        Default: sf_start
    sf_trunc_key : str, optional
        Truncation time of the SFH, in Gyr. Only used if sfh=1 or sfh=4 or sfh=5,
        dataset keyword name
        Default: sf_trunc
    stellar_ages_key : str, optional
        galaxy stellar ages (Gyr),dataset keyword name
        Default: stellar_age
    fburst_key : str, optional
        Deﬁnes the fraction of mass formed in an instantaneous burst of star formation.
        Only used if sfh=1 or sfh=4,dataset keyword name
        Default: fburst
    tburst_key : str, optional
        Defines the age of the Universe when the burst occurs. If tburst > tage then
        there is no burst. Only used if sfh=1 or sfh=4, dataset keyword name
        Default: tburst
    sf_slope_key : str, optional
        For sfh=5, this is the slope of the SFR after time sf_trunc, dataset keyword
        name
        Default: sf_slope
    dust_type : int, optional
        attenuation curve for dust type, see FSPS manual, default Calzetti
        Default: 2
    dust_tesc : float, optional
        Stars younger than dust_tesc are attenuated by both dust1 and dust2, while stars
        older are attenuated by dust2 only. Units are log(yrs)
        Default: 7.0
    dust_birth_cloud_key : str, optional
        dust parameter describing young stellar light attenuation (dust1 in FSPS),
        dataset keyword name
        Default: dust1_birth_cloud
    dust_diffuse_key : str, optional
        dust parameters describing old stellar light attenuation (dust2 in FSPS) dataset
        keyword name
        Default: dust2_diffuse
    dust_clumps : int, optional
        Dust parameter describing the dispersion of a Gaussian PDF density distribution
        for the old dust. Setting this value to -99.0 sets the distribution to a uniform
        screen, values other than -99 are no longer supported
        Default: -99
    frac_nodust : float, optional
        Fraction of starlight that is not attenuated by the diffuse dust component
        Default: 0.0
    frac_obrun : float, optional
        Fraction of the young stars (age < dust_tesc) that are not attenuated by dust1
        and that do not contribute to any nebular emission, representing runaway OB
        stars or escaping ionizing radiation. These stars are still attenuated by dust2.
        Default: 0.0
    dust_index_key : str, optional
        Power law index of the attenuation curve. Only used when dust_type=0, dataset
        keyword name
        Default: dust_index
    dust_powerlaw_modifier_key : str, optional
        power-law modifiers to the shape of the Calzetti et al. (2000) attenuation curve
        (dust1_index),dataset keyword name
        Default: dust_calzetti_modifier
    mwr_key : str, optional
        The ratio of total to selective absorption which characterizes the MW extinction
        curve: RV=AV/E(B-V), used when dust_type=1,dataset keyword name
        Default: mwr
    uvb_key : str, optional
        Parameter characterizing the strength of the 2175A extinction feature with
        respect to the standard Cardelli et al. determination for the MW. Only used when
        dust_type=1,dataset keyword name
        Default: uvb
    wgp1_key : str, optional
        Integer specifying the optical depth in the Witt & Gordon (2000) models. Values
        range from 1 − 18, used only whendust_type=3, dataset keyword name
        Default: wgp1
    wgp2 : int, optional
        Integer specifying the type of large-scale geometry and extinction curve. Values
        range from 1-6, used only when dust_type=3
        Default: 1
    wgp3 : int, optional
        Integer specifying the local geometry for the Witt & Gordon (2000) dust models,
        used only when dust_type=3
        Default: 1
    dust_emission_gamma_key : str, optional
        Relative contributions of dust heated at Umin, parameter of Draine and Li (2007)
        dust emission modeldataset keyword name
        Default: dust_gamma
    dust_emission_umin_key : str, optional
        Minimum radiation field strengths, parameter of Draine and Li (2007) dust
        emission model, dataset keyword name
        Default: dust_umin
    dust_emission_qpah_key : str, optional
        Grain size distributions in mass in PAHs, parameter of Draine and Li (2007) dust
        emission model,dataset keyword name
        Default: dust_qpah
    fraction_agn_bol_lum_key : str, optional
        Fractional contributions of AGN wrt stellar bolometric luminosity, dataset
        keyword name
        Default: f_agn
    agn_torus_opt_depth_key : str, optional
        Optical depths of the AGN dust torii dataset keyword name
        Default: tau_agn
    tabulated_sfh_key : str, optional
        tabulated SFH dataset keyword name
        Default: tabulated_sfh
    tabulated_lsf_key : str, optional
        tabulated LSF dataset keyword name
        Default: tabulated_lsf
    physical_units : bool, optional
        A parameter
        Default: False
    msg : unknown type, optional
        Default: False (True) for rest-frame spectra in units ofLsun/Hz (erg/s/Hz)
    restframe_wave_key : str, optional
        Rest-frame wavelength keyword name of the output hdf5 dataset
        Default: restframe_wavelengths
    restframe_sed_key : str, optional
        Rest-frame SED keyword name of the output hdf5 dataset
        Default: restframe_seds

    Returns
    -------
    dict
        ModelHandle storing the rest-frame SED models
    """
