import os

import ceci
import numpy as np
import pandas as pd
import pytest
import pzflow

from rail.creation.degraders.photometric_errors import LSSTErrorModel
from rail.creation.engines.flowEngine import FlowCreator
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer


def test_creator():
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )

    n_samples = int(1e5)
    flowCreator_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )
    samples_truth = flowCreator_truth.sample(n_samples, seed=0)
    errorModel = LSSTErrorModel.make_stage(name="error_model", input="output_truth.pq")
    test_data = errorModel(samples_truth)
