#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.estimation.algos.train_z import TrainZInformer, TrainZEstimator
from rail.evaluation.single_evaluator import SingleEvaluator

from rail.utils.name_utils import NameFactory
from rail.core.stage import RailStage, RailPipeline

import ceci



input_file = 'rubin_dm_dc2_example.pq'


class TrainZPipeline(RailPipeline):

    default_input_dict = dict(
        input_train='dummy.in',
        input_test='dummy.in',
    )

    def __init__(self, namer, selection='default', flavor='baseline'):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        path_kwargs = dict(
            selection=selection,
            algorithm='train_z',
            flavor=flavor,
        )
        
        self.inform_trainz = TrainZInformer.build(
            aliases=dict(input='input_train'),
            model=namer.resolve_path_template(
                'estimator_model_path',
                model_suffix='pkl',
                **path_kwargs,
            ),
            hdf5_groupname='',
        )

        self.estimate_trainz = TrainZEstimator.build(
            aliases=dict(input='input_test'),
            connections=dict(
                model=self.inform_trainz.io.model,
            ),
            output=namer.resolve_path_template(
                'pz_pdf_path',
                **path_kwargs,
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
            output=namer.resolve_path_template(
                'per_object_metrics_path', 
                **path_kwargs,
            ),
            summary=namer.resolve_path_template(
                'summary_value_metrics_path',
                **path_kwargs,
            ),
            single_distribution_summary=namer.resolve_path_template(
                'summary_pdf_metrics_path',
                **path_kwargs,
            ),
            hdf5_groupname='',
        )
