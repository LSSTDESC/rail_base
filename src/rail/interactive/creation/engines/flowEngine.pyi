from typing import Any

def flow_creator(**kwargs) -> Any:
    """
    Creator wrapper for a PZFlow Flow object.

    ---

    Draw samples from the model specified in the configuration.

    This is a method for running a Creator in interactive mode. In pipeline
    mode, the subclass ``run`` method will be called by itself.

    ---

    This function was generated from the function
    rail.creation.engines.flowEngine.FlowCreator.sample

    Parameters
    ----------
    n_samples : int, required
        The number of samples to draw, by default None
        Number of samples to create
    model : FlowHandle, required
    seed : int, optional
        Random number seed
        Default: 12345
        The random seed to control sampling, by default None

    Returns
    -------
    A tablesio-compatible table
        TableHandle wrapping the newly created samples


    Notes
    -----
    This method puts ``n_samples`` and ``seed`` into the stage configuration
    data, which makes them available to other methods.

    It then calls the ``run`` method, which must be defined by a subclass.

    Finally, the ``TableHandle`` associated to the ``output`` tag is returned.
    """

def flow_modeler(**kwargs) -> Any:
    """
    Modeler wrapper for a PZFlow Flow object.

    This class trains the flow.

    ---

    Produce a creation model from which photometry and redshifts can be
    generated.

    ---

    This function was generated from the function
    rail.creation.engines.flowEngine.FlowModeler.fit_model

    Parameters
    ----------
    input_data : Data for RAIL, required
        ???
    seed : int, optional
        The random seed for training.
        Default: 0
    phys_cols : dict, optional
        Names of non-photometry columns and their corresponding [min, max] values.
        Default: {'redshift': [0, 3]}
    phot_cols : dict, optional
        Names of photometry columns and their corresponding [min, max] values.
        Default: {'mag_u_lsst': [17, 35], 'mag_g_lsst': [16, 32], 'mag_r_lsst':...}
    calc_colors : dict, optional
        Whether to internally calculate colors (if phot_cols are magnitudes). Assumes
        that you want to calculate colors from adjacent columns in phot_cols. If you do
        not want to calculate colors, set False. Else, provide a dictionary
        {'ref_column_name': band}, where band is a string corresponding to the column in
        phot_cols you want to save as the overall galaxy magnitude.
        Default: {'ref_column_name': 'mag_i_lsst'}
    spline_knots : int, optional
        The number of spline knots in the normalizing flow.
        Default: 16
    num_training_epochs : int, optional
        The number of training epochs.
        Default: 30

    Returns
    -------
    numpy.ndarray
        This will definitely be a wrapper around a File,
        but the filetype and format depend entirely on the
        modeling approach
    """

def flow_posterior(**kwargs) -> Any:
    """
    PosteriorCalculator wrapper for a PZFlow Flow object

    .. code-block:: text

        data : pd.DataFrame
            Pandas dataframe of the data on which the posteriors are conditioned.
            Must have all columns in self.flow.data_columns, *except*
            for the column specified for the posterior (see below).

        column : str
            Name of the column for which the posterior is calculated.
            Must be one of the columns in self.flow.data_columns. However,
            whether or not this column is present in `data` is irrelevant.

        grid : np.ndarray
            Grid over which the posterior is calculated.

        err_samples : int, optional
            Number of samples from the error distribution to average over for
            the posterior calculation. If provided, Gaussian errors are assumed,
            and method will look for error columns in `inputs`. Error columns
            must end in `_err`. E.g. the error column for the variable `u` must
            be `u_err`. Zero error assumed for any missing error columns.

        seed: int, optional
            Random seed for drawing samples from the error distribution.

        marg_rules : dict, optional
            Dictionary with rules for marginalizing over missing variables.
            The dictionary must contain the key "flag", which gives the flag
            that indicates a missing value. E.g. if missing values are given
            the value 99, the dictionary should contain {"flag": 99}.
            The dictionary must also contain {"name": callable} for any
            variables that will need to be marginalized over, where name is
            the name of the variable, and callable is a callable that takes
            the row of variables and returns a grid over which to marginalize
            the variable. E.g. {"y": lambda row: np.linspace(0, row["x"], 10)}.
            Note: the callable for a given name must *always* return an array
            of the same length, regardless of the input row.
            DEFAULT: the default marg_rules dict is
            {"flag": np.nan,
            "u": np.linspace(25, 31, 10),}

        batch_size: int, default=None
            Size of batches in which to calculate posteriors. If None, all
            posteriors are calculated simultaneously. This is faster, but
            requires more memory.

        nan_to_zero : bool, default=True
            Whether to convert NaN's to zero probability in the final pdfs.

    ---

    Return posteriors for the given column.

    This is a method for running a Creator in interactive mode. In pipeline
    mode, the subclass ``run`` method will be called by itself.

    ---

    This function was generated from the function
    rail.creation.engines.flowEngine.FlowPosterior.get_posterior

    Parameters
    ----------
    input_data : TableLike, required
        A table of the galaxies for which posteriors are calculated
    column : str, required
        Column to compute posterior for
    model : FlowHandle, required
    grid : list, optional
        Grid over which the posterior is calculated
        Default: []
    err_samples : int, optional
        A parameter
        Default: 10
    seed : int, optional
        A parameter
        Default: 12345
    marg_rules : dict, optional
        A parameter
        Default: {'flag': nan, 'mag_u_lsst': <function FlowPosterior.<lambda> at...}
    batch_size : unknown type, optional
        Default: 10000
    nan_to_zero : unknown type, optional
        Default: True

    Returns
    -------
    qp.core.ensemble.Ensemble
        Posterior Estimate


    Notes
    -----
    This will put the ``data`` argument input this Stages the DataStore
    using this stages ``input`` tag.

    This will put the additional functional arguments into this Stages
    configuration data.

    It will then call ``self.run()`` and return the ``QPHandle``
    associated to the ``output`` tag.
    """
