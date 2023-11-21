"""
Abstract base classes for Informers.
These superstages ingest prior information, including training sets and explicit
priors, and prepare a model that can be used to produce photo-z data products.
They are distinguished by their input data types, and the models they output can
be used for their corresponding Estimator, Summarizer, or Classifier stages.
"""

from rail.core.data import TableHandle, QPHandle, ModelHandle
from rail.core.stage import RailStage


class CatInformer(RailStage):
    """The base class for informing models used to make photo-z data products
    from catalog-like inputs (i.e., tables with fluxes in photometric bands among
    the set of columns).

    Estimators use a generic "model", the details of which depends on the sub-class.
    Most estimators will have associated Informer classes, which can be used to inform
    those models.

    (Note, "Inform" is more generic than "Train" as it also applies to algorithms that
    are template-based rather than machine learning-based.)

    Informer will produce as output a generic "model", the details of which depends on the sub-class.

    They take as "input" catalog-like tabular data, which is used to "inform" the model.
    """

    name = "CatInformer"
    config_options = RailStage.config_options.copy()
    inputs = [("input", TableHandle)]
    outputs = [("model", ModelHandle)]

    def __init__(self, args, comm=None):
        """Initialize Informer that can inform models for redshift estimation"""
        RailStage.__init__(self, args, comm=comm)
        self.model = None

    def inform(self, training_data):
        """The main interface method for Informers

        This will attach the input_data to this `Informer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the model that it creates to this Estimator
        by using `self.add_data('model', model)`.

        Finally, this will return a ModelHandle providing access to the trained model.

        Parameters
        ----------
        input_data : `dict` or `TableHandle`
            dictionary of all input data, or a `TableHandle` providing access to it

        Returns
        -------
        model : ModelHandle
            Handle providing access to trained model
        """
        self.set_data("input", training_data)
        self.run()
        self.finalize()
        return self.get_handle("model")


class PzInformer(RailStage):
    """The base class for informing models used to make photo-z data products from
    existing ensembles of p(z) distributions.

    PzSummarizers can use a generic "model", the details of which depends on the sub-class.
    Some summaer will have associated PzInformer classes, which can be used to inform
    those models.

    (Note, "Inform" is more generic than "Train" as it also applies to algorithms that
    are template-based rather than machine learning-based.)

    PzInformer will produce as output a generic "model", the details of which depends on the sub-class.

    They take as "input" a qp.Ensemble of per-galaxy p(z) data, which is used to "inform" the model.
    """

    name = "PzInformer"
    config_options = RailStage.config_options.copy()
    inputs = [("input", QPHandle)]
    outputs = [("model", ModelHandle)]

    def __init__(self, args, comm=None):
        """Initialize Informer that can inform models for redshift estimation"""
        RailStage.__init__(self, args, comm=comm)
        self.model = None

    def inform(self, training_data):
        """The main interface method for Informers

        This will attach the input_data to this `Informer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the model that it creates to this Estimator
        by using `self.add_data('model', model)`.

        Finally, this will return a ModelHandle providing access to the trained model.

        Parameters
        ----------
        input_data :  `qp.Ensemble`
            Per-galaxy p(z), and any ancilary data associated with it

        Returns
        -------
        model : ModelHandle
            Handle providing access to trained model
        """
        self.set_data("input", training_data)
        self.run()
        self.finalize()
        return self.get_handle("model")
