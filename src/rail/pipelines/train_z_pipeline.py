#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.estimation.algos.train_z import TrainZInformer, TrainZEstimator

from rail.utils.name_utils import NameFactory, DataType, CatalogType, ModelType, PdfType
from rail.core.stage import RailStage, RailPipeline

import ceci


namer = NameFactory()

input_file = 'rubin_dm_dc2_example.pq'


class TrainZPipeline(RailPipeline):

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        self.inform_trainz = TrainZInformer.build(
            aliases=dict(input='input_train'),
            model=os.path.join(namer.get_data_dir(DataType.model, ModelType.estimator), "model_trainz.pkl"),            
            hdf5_groupname='',
        )

        self.estimate_trainz = TrainZEstimator.build(
            aliases=dict(input='input_test'),
            connections=dict(
                model=self.inform_trainz.io.model,
            ),
            output=os.path.join(namer.get_data_dir(DataType.pdf, PdfType.pz), "output_trainz.hdf5"),
            hdf5_groupname='',
        )


if __name__ == '__main__':    
    pipe = TrainZPipeline()
    input_dict = dict(
        input_train='dummy.in',
        input_test='dummy.in',
    )
    pipe.initialize(input_dict, dict(output_dir='.', log_dir='.', resume=False), None)
    pipe.save('tmp_train_z.yml')
