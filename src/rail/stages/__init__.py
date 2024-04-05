import rail
from rail.core import RailEnv

from rail.estimation.estimator import *
from rail.estimation.classifier import *
from rail.estimation.summarizer import *
from rail.estimation.algos.naive_stack import *
from rail.estimation.algos.random_gauss import *
from rail.estimation.algos.point_est_hist import *
from rail.estimation.algos.train_z import *
from rail.estimation.algos.var_inf import *
from rail.estimation.algos.uniform_binning import *
from rail.estimation.algos.equal_count import *


from rail.creation.degrader import *
#from rail.creation.degradation.spectroscopic_degraders import *
# from rail.creation.degradation.spectroscopic_selections import *
from rail.creation.degradation.quantityCut import *

from rail.creation.engine import *

# from rail.creation.engines.flowEngine import *
# from rail.creation.engines.galaxy_population_components import *
# from rail.creation.engines.dsps_photometry_creator import *
# from rail.creation.engines.dsps_sed_modeler import *

# from rail.evaluation.evaluator import Evaluator


def import_and_attach_all():
    """Import all the packages in the rail ecosystem and attach them to this module"""
    RailEnv.import_all_packages()
    RailEnv.attach_stages(rail.stages)
