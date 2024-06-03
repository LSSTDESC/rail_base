#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.estimation.algos.train_z import TrainZInformer, TrainZEstimator
from rail.evaluation.single_evaluator import SingleEvaluator

from rail.utils.name_utils import NameFactory, DataType, CatalogType, ModelType, PdfType, MetricType
from rail.core.stage import RailStage, RailPipeline

import ceci


namer = NameFactory()

input_file = 'rubin_dm_dc2_example.pq'


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
            model=os.path.join(namer.get_data_dir(DataType.models, ModelType.estimator), "model_trainz.pkl"),            
            hdf5_groupname='',
        )

        self.estimate_trainz = TrainZEstimator.build(
            aliases=dict(input='input_test'),
            connections=dict(
                model=self.inform_trainz.io.model,
            ),
            output=os.path.join(namer.get_data_dir(DataType.pdfs, PdfType.pz), "output_trainz.hdf5"),
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
            output=os.path.join(
                namer.get_data_dir(DataType.metrics, MetricType.per_object),
                "output_trainz.hdf5",
            ),
            summary=os.path.join(
                namer.get_data_dir(DataType.metrics, MetricType.summary_value),
                "summary_trainz.hdf5"
            ),
            single_distribution_summary=os.path.join(
                namer.get_data_dir(DataType.metrics, MetricType.summary_pdf),
                "single_distribution_summary_trainz.hdf5"
            ),
            hdf5_groupname='',
        )
