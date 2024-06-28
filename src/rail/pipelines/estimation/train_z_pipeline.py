#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.estimation.algos.train_z import TrainZInformer, TrainZEstimator
from rail.evaluation.single_evaluator import SingleEvaluator

from rail.core.stage import RailStage, RailPipeline

import ceci


class TrainZPipeline(RailPipeline):

    default_input_dict = dict(
        input_train='dummy.in',
        input_test='dummy.in',
    )

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        
        self.inform_trainz = TrainZInformer.build(
            aliases=dict(input='input_train'),
            hdf5_groupname='',
        )

        self.estimate_trainz = TrainZEstimator.build(
            aliases=dict(input='input_test'),
            connections=dict(
                model=self.inform_trainz.io.model,
            ),
            hdf5_groupname='',
        )

        self.evalute_trainz = SingleEvaluator.build(
            aliases=dict(truth='input_test'),
            connections=dict(
                input=self.estimate_trainz.io.output,
            ),
            point_estimates=['mode'],
            truth_point_estimates=["redshift"],
            metrics=["all"],
            metric_config=dict(brier=dict(limits=[0., 3.5])),
            exclude_metrics=['rmse', 'ks', 'kld', 'cvm', 'ad', 'rbpe', 'outlier'],
            hdf5_groupname='',
        )
