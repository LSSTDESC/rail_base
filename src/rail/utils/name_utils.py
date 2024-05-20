"""
Utility code to help define standard paths for various data products
"""

import os
import enum


class DataType(enum.Enum):

    catalog = 0
    model = 1
    config = 2
    pdf = 3
    metric = 4


class CatalogType(enum.Enum):

    reference = 0
    created = 1
    degraded = 2


class ModelType(enum.Enum):

    creator = 0
    degrarder = 1
    estimator = 2
    summarizer = 3
    evaluator = 4


class PdfType(enum.Enum):

    pz = 0
    nz = 1


class MetricType(enum.Enum):

    pz = 0
    nz = 1


class NameFactory:

    project_directory_template = os.path.join(
        "{root}",
        "{project}",
        "{study}",
    )

    data_directory_template = os.path.join(
        "{data_type}",
        "{data_subtype}",
    )
    
    full_directory_template = os.path.join(
        project_directory_template,
        data_directory_template
    )

    
    def get_project_dir(self, root, project, study):
        return self.project_directory_template.format(root=root, project=project, study=study)

    def get_data_dir(self, data_type, data_subtype):
        return self.data_directory_template.format(data_type=data_type.name, data_subtype=data_subtype.name)

