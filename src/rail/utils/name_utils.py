"""
Utility code to help define standard paths for various data products
"""

import os
import enum


class DataType(enum.Enum):

    pipelines = 0
    catalogs = 1
    models = 2
    pdfs = 3
    metrics = 4


class PipelineType(enum.Enum):

    inform = 0
    estimate = 1
    evaluate = 2
    summarize = 3
    

class CatalogType(enum.Enum):

    truth = 0
    truth_reduced = 1
    observed = 2


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

    per_object = 0
    summary_value = 1
    summary_pdf = 2
    

class NameFactory:

    project_directory_template = os.path.join(
        "{root}",
        "projects",
        "{project}",
    )

    data_directory_template = os.path.join(
        "{data_type}",
        "{data_subtype}",        
    )
    
    full_directory_template = os.path.join(
        project_directory_template,
        data_directory_template
    )
    
    def get_project_dir(self, root, project):
        return self.project_directory_template.format(root=root, project=project)

    def get_data_dir(self, data_type, data_subtype):
        if data_subtype is None:
            return f"{data_type}"
        return self.data_directory_template.format(data_type=data_type.name, data_subtype=data_subtype.name)

    def get_full_dir(self, root, project, data_type, data_subtype):
        return os.path.join(
            self.get_project_dir(root, project),
            self.get_data_dir(data_type, data_subtype),
        )
