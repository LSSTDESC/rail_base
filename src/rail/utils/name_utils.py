"""
Utility code to help define standard paths for various data products
"""

import copy
# import enum
import os
from functools import partial


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


def _resolve_dict(source, interpolants):
    """
    Recursively resolve a dictionary using inteprolants
    """
    if source is not None:
        sink = copy.deepcopy(source)
        for k, v in source.items():
            match v:
                case dict():
                    v_interpolated = _resolve_dict(source[k], interpolants)
                case list():
                    v_interpolated = [_resolve_dict(_v, interpolants) for _v in v]
                case str():
                    v_interpolated = v.format(**interpolants)
                case _:
                    raise ValueError("Cannot interpolate type!")

            sink[k] = v_interpolated
    else:
        sink = None

    return sink

def _resolve(templates, source, interpolants):
    sink = copy.deepcopy(templates)
    if (overrides := source) is not None:
        for k, v in overrides.items():
            sink[k] = v
    for k, v in sink.items():
        match v:
            case partial():
                sink[k] = v(**sink)
            case _:
                continue
    sink = _resolve_dict(sink, interpolants)
    return sink


class NameFactory:

    def __init__(self, config={}, templates={}, interpolants={}):

        self._config = copy.deepcopy(config)
        self._templates = copy.deepcopy(templates)
        self._interpolants = {}

        self.templates = {}
        for k, v in templates.items():
            self.templates[k] = partial(v.format, **templates)
        # if (common := self._config.get("CommonPaths")) is not None:
        #     for k, v in self.templates.items():
        #         self.templates[k] = v(**self.templates)
        # self.templates["project"] = self.project

        self.interpolants = interpolants

    @property
    def interpolants(self):
        return self._interpolants

    @interpolants.setter
    def interpolants(self, config):
        for key, value in config.items():
            new_value = value.format(**self.interpolants)
            self.interpolants[key] = new_value

    @interpolants.deleter
    def interpolants(self):
        self._interpolants = {}

    # def get_interpolants(self, config):
    #     interpolants = {}

    #     for key, value in config.items():
    #         new_value = value.format(**interpolants)
    #         interpolants[key] = new_value

    #     return interpolants

    def resolve_from_config(self, config):
        # interpolants = self.get_interpolants(config)
        # self.interpolants = config
        resolved = _resolve(
            self.templates,
            config,
            self.interpolants,
        )
        config.update(resolved)

        return resolved

    def resolve_path(self, config, path_key, **kwargs):
        if (path_value := config.get(path_key)) is not None:
            formatted = path_value.format(**kwargs, **self.interpolants)
        else:
            raise ValueError(f"Path '{path_key}' not found in {config}")
        return formatted

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
