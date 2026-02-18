from typing import Any

def random_forest_classifier(**kwargs) -> Any:
    """
    Classifier that assigns tomographic bins based on random forest method

    ---

    The main run method for the classifier, should be implemented
    in the specific subclass.

    This will attach the input_data to this `CatClassifier`
    (for introspection and provenance tracking).

    Then it will call the run() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the data that it creates to this Classifier
    by using `self.add_data('output', output_data)`.

    Finally, this will return a TableHandle providing access to that output data.

    ---

    This function was generated from the function
    rail.estimation.algos.random_forest.RandomForestClassifier.classify

    Parameters
    ----------
    input_data : TableLike, required
        A dictionary of all input data
    model : numpy.ndarray, required
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    id_name : str, optional
        Column name for the object ID in the input data, if empty the row index is used
        as the ID.
        Default:
    class_bands : list, optional
        Which bands to use for classification
        Default: ['r', 'i', 'z']
    bands : dict, optional
        column names for the the bands
        Default: {'r': 'mag_r_lsst', 'i': 'mag_i_lsst', 'z': 'mag_z_lsst'}

    Returns
    -------
    A tablesio-compatible table
        Class assignment for each galaxy.
    """

def random_forest_informer(**kwargs) -> Any:
    """
    Train the random forest classifier

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
    rail.estimation.algos.random_forest.RandomForestInformer.inform

    Parameters
    ----------
    training_data : TableLike, required
        dictionary of all input data, or a `TableHandle` providing access to it
    random_seed : int, required
        random seed
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    class_bands : list, optional
        Which bands to use for classification
        Default: ['r', 'i', 'z']
    bands : dict, optional
        column names for the the bands
        Default: {'r': 'mag_r_lsst', 'i': 'mag_i_lsst', 'z': 'mag_z_lsst'}
    redshift_col : str, optional
        Redshift column names
        Default: sz
    bin_edges : list, optional
        Binning for training data
        Default: [0, 0.5, 1.0]
    no_assign : int, optional
        Value for no assignment flag
        Default: -99

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
