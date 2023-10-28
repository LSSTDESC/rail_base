import os

from rail.core.data import QPHandle
from rail.core.stage import RailStage
from rail.core.utils import RAILDIR
from rail.estimation.algos import naive_stack, point_est_hist, var_inf

testdata = os.path.join(RAILDIR, "rail/examples_data/testdata/output_BPZ_lite.hdf5")
DS = RailStage.data_store


def one_algo(key, summarizer_class, summary_kwargs):
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    DS.__class__.allow_overwrite = True
    test_data = DS.read_file("test_data", QPHandle, testdata)
    summarizer = summarizer_class.make_stage(name=key, **summary_kwargs)
    summary_ens = summarizer.summarize(test_data)
    os.remove(summarizer.get_output(summarizer.get_aliased_tag("output"), final_name=True))
    os.remove(summarizer.get_output(summarizer.get_aliased_tag("single_NZ"), final_name=True))
    return summary_ens


def test_naive_stack():
    """Basic end to end test for the Naive stack informer to estimator stages
    """
    naive_stack_informer_stage = naive_stack.NaiveStackInformer.make_stage()
    naive_stack_informer_stage.inform('')

    summary_config_dict = {}
    summarizer_class = naive_stack.NaiveStackSummarizer
    _ = one_algo("NaiveStack", summarizer_class, summary_config_dict)


def test_point_estimate_hist():
    """Basic end to end test for the point estimate histogram informer to estimator
    stages
    """
    point_est_informer_stage = point_est_hist.PointEstHistInformer.make_stage()
    point_est_informer_stage.inform('')

    summary_config_dict = {}
    summarizer_class = point_est_hist.PointEstHistSummarizer
    _ = one_algo("PointEstimateHist", summarizer_class, summary_config_dict)


def test_var_inference_stack():
    """Basic end to end test for the var inference informer to estimator stages
    """
    var_inf_informer_stage = var_inf.VarInfStackInformer.make_stage()
    var_inf_informer_stage.inform('')

    summary_config_dict = {}
    summarizer_class = var_inf.VarInfStackSummarizer
    _ = one_algo("VariationalInference", summarizer_class, summary_config_dict)
