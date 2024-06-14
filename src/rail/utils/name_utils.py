"""
Utility code to help define standard paths for various data products
"""

import copy
# import enum
import os
from functools import partial


def resolve_dict(source, interpolants):
    """
    Recursively resolve a dictionary using inteprolants
    """
    if source is not None:
        sink = copy.deepcopy(source)
        for k, v in source.items():
            match v:
                case dict():
                    v_interpolated = resolve_dict(source[k], interpolants)
                case list():
                    v_interpolated = [resolve_dict(_v, interpolants) for _v in v]
                case str():
                    v_interpolated = v.format(**interpolants)
                case _:
                    raise ValueError("Cannot interpolate type!")

            sink[k] = v_interpolated
    else:
        sink = None

    return sink


# class DataType(enum.Enum):
# 
#     pipelines = 0
#     catalogs = 1
#     models = 2
#     pdfs = 3
#     metrics = 4


# class PipelineType(enum.Enum):
# 
#     inform = 0
#     estimate = 1
#     evaluate = 2
#     summarize = 3


# class CatalogType(enum.Enum):
# 
#     truth = 0
#     truth_reduced = 1
#     degraded = 2
#     observed = 3


# class ModelType(enum.Enum):
# 
#     creator = 0
#     degrarder = 1
#     estimator = 2
#     summarizer = 3
#     evaluator = 4


# class PdfType(enum.Enum):
# 
#     pz = 0
#     nz = 1


# class MetricType(enum.Enum):
# 
#     per_object = 0
#     summary_value = 1
#     summary_pdf = 2


class NameFactory:

    defaults = {
        "root": os.getcwd(),
        "project_root": os.path.join("{root}"),
        "data_root": os.path.join("{root}"),
        "project_dir": os.path.join(
            "{project_root}",
            "{project}",
        ),
        "project_data_dir": os.path.join(
            "{data_root}",
            "{project}",
        ),
        "catalogs_path": os.path.join(
            "{project_data_dir}",
            "catalogs",
        ),
        "pipelines_path": os.path.join(
            "{project_dir}",
            "pipelines",
        ),
        "models_path": os.path.join(
            "{project_data_dir}",
            "models",
        ),
        "pdfs_path": os.path.join(
            "{project_data_dir}",
            "pdfs",
        ),
        "metrics_path": os.path.join(
            "{project_data_dir}",
            "metrics",
        ),
    }

    def __init__(self, project, config={}):
        self.project = project

        self._config = copy.deepcopy(config)
        for k, v in self.defaults.items():
            self.defaults[k] = partial(v.format, **self.defaults)
        # if (common := self._config.get("CommonPaths")) is not None:
        #     for k, v in self.defaults.items():
        #         self.defaults[k] = v(**self.defaults)
        # self.defaults["project"] = self.project

    def resolve(self, source, interpolants):
        sink = copy.deepcopy(self.defaults)
        sink["project"] = self.project
        if (overrides := source) is not None:
            for k, v in overrides.items():
                sink[k] = v
        for k, v in sink.items():
            match v:
                case partial():
                    sink[k] = v(**sink)
                case _:
                    continue
        sink = resolve_dict(sink, interpolants)
        return sink

    # def get_project_dir(self, root, project):
    #     return self.project_directory_template.format(root=root, project=project)

    # def get_data_dir(self, data_type, data_subtype):
    #     if data_subtype is None:
    #         return f"{data_type}"
    #     return self.data_directory_template.format(data_type=data_type.name, data_subtype=data_subtype.name)

    # def get_full_dir(self, root, project, data_type, data_subtype):
    #     return os.path.join(
    #         self.get_project_dir(root, project),
    #         self.get_data_dir(data_type, data_subtype),
    #     )
