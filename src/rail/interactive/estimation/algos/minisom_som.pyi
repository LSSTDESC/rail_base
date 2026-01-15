from typing import Any

def mini_som_informer(**kwargs) -> Any:
    """
    Summarizer that uses a SOM to construct a weighted sum
    of spec-z objects in the same SOM cell as each photometric
    galaxy in order to estimate the overall N(z).  This is
    very related to the NZDir estimator, though that estimator
    actually reverses this process and looks for photometric
    neighbors around each spectroscopic galaxy, which can
    lead to problems if there are photometric galaxies with
    no nearby spec-z objects (NZDir is not aware that such
    objects exist and thus can hid biases).
    Part of the SimpeSOM estimator will be a check for cells
    which contain photometric objects but do not contain any
    corresponding training/spec-z objects, those unmatched
    objects will be flagged for possible removal from the
    input sample.
    The inform stage will simply construct a 2D grid SOM
    using `minisom` from a large sample of input
    photometric data and save this as an output.  This may
    be a computationally intensive stage, though it will
    hopefully be run once and used by the estimate/summarize
    stage many times without needing to be re-run.

    We can make the SOM either with all colors, or one
    magnitude and N colors, or an arbitrary set of columns.
    The code includes a flag `column_usage` to set usage,
    If set to "colors" it will take the difference of each
    adjacen pair of columns in `bands` as the colors. If
    set to `magandcolors` it will use these colors plus one
    magnitude as specified by `ref_band`.  If set to
    `columns` then it will take as inputs all of the columns
    specified by `bands` (they can be magnitudes, colors,
    or any other input specified by the user).  NOTE: any
    custom `bands` parameters must have an accompanying
    `nondetect_val` dictionary that will replace
    nondetections with the nondetect_val values!

    This will make a pickle file containing the `minisom` SOM object that
    will be used by the estimation/summarization stage

    ---

    The main interface method for Informers

    This will attach the input_data to this `Informer`
    (for introspection and provenance tracking).

    Then it will call the run(), validate() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the model that it creates to this Estimator
    by using `self.add_data('model', model)`.

    Finally, this will return a ModelHandle providing access to the trained model.

    ---

    This function was generated from the function
    rail.estimation.algos.minisom_som.MiniSOMInformer.inform

    Parameters
    ----------
    input : TableLike
        dictionary of all input data, or a `TableHandle` providing access to it
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    mag_limits : dict, optional
        Limiting magnitudes by filter
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    bands : list, optional
        Names of columns for magnitude by filter band
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    ref_band : str, optional
        band to use in addition to colors
        Default: mag_i_lsst
    column_usage : str, optional
        switch for how SOM uses columns, valid values are 'colors', 'magandcolors', and
        'columns'
        Default: magandcolors
    seed : int, optional
        Random number seed
        Default: 0
    m_dim : int, optional
        number of cells in SOM y dimension
        Default: 31
    n_dim : int, optional
        number of cells in SOM x dimension
        Default: 31
    som_sigma : float, optional
        sigma param in SOM training
        Default: 1.5
    som_learning_rate : float, optional
        SOM learning rate
        Default: 0.5
    som_iterations : int, optional
        number of iterations in SOM training
        Default: 10000

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """

def mini_som_summarizer(**kwargs) -> Any:
    """
    Quick implementation of a SOM-based summarizer that
    constructs and N(z) estimate via a weighted sum of the
    empirical N(z) consisting of the normalized histogram
    of spec-z values contained in the same SOM cell as
    each photometric galaxy.
    There are some general guidelines to choosing the geometry
    and number of total cells in the SOM.  This paper:
    http://www.giscience2010.org/pdfs/paper_230.pdf
    recommends 5*sqrt(num rows * num data columns) as a rough
    guideline.  Some authors state that a SOM with one
    dimension roughly twice as long as the other are better,
    while others find that square SOMs with equal X and Y
    dimensions are best, the user can set the dimensions
    using the n_dim and m_dim parameters.
    For more discussion on SOMs and photo-z calibration, see
    the KiDS paper on the topic:
    http://arxiv.org/abs/1909.09632
    particularly the appendices.
    Note that several parameters are stored in the model file,
    e.g. the columns used. This ensures that the same columns
    used in constructing the SOM are used when finding the
    winning SOM cell with the test data.
    Two additional files are also written out:
    `cellid_output` outputs the 'winning' SOM cell for each
    photometric galaxy, in both raveled and 2D SOM cell
    coordinates.  If the objectID or galaxy_id is present
    they will also be included in this file, if not the
    coordinates will be written in the same order in which
    the data is read in.
    `uncovered_cell_file` outputs the raveled cell
    IDs of cells that contain photometric
    galaxies but no corresponding spectroscopic objects,
    these objects should be removed from the sample as they
    cannot be accounted for properly in the summarizer.
    Some iteration on data cuts may be necessary to
    remove/mitigate these 'uncovered' objects.

    ---

    The main run method for the summarization, should be implemented
    in the specific subclass.

    This will attach the input_data to this `SZandPhottoNZSummarizer`
    (for introspection and provenance tracking).

    Then it will call the run() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the data that it creates to this Estimator
    by using `self.add_data('output', output_data)`.

    Finally, this will return a QPHandle providing access to that output data.

    ---

    This function was generated from the function
    rail.estimation.algos.minisom_som.MiniSOMSummarizer.summarize

    Parameters
    ----------
    input : dict["input_data": qp.Ensemble, "spec_data": np.ndarray]
        Dictionary of input data with the following keys:
        input_data: qp.Ensemble - Per-galaxy p(z), and any ancillary data associated
        with it
        spec_data: np.ndarray - Spectroscopic data
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    mag_limits : dict, optional
        Limiting magnitudes by filter
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    objid_name : str, optional
        A parameter
        Default:
    spec_groupname : str, optional
        name of hdf5 group for spec data, if None, then set to ''
        Default: photometry
    seed : int, optional
        random seed
        Default: 12345
    phot_weightcol : str, optional
        name of photometry weight, if present
        Default:
    spec_weightcol : str, optional
        name of specz weight col, if present
        Default:
    nsamples : int, optional
        number of bootstrap samples to generate
        Default: 20

    Returns
    -------
    qp.Ensemble
        Ensemble with n(z), and any ancillary data
    """
